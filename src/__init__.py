from ovos_utils import classproperty
from ovos_utils.process_utils import RuntimeRequirements
from ovos_workshop.decorators import intent_handler
from ovos_workshop.skills import OVOSSkill

import os
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer


# NTR data and tuning parameters in <NTR_Skill>/settings.json
DEFAULT_SETTINGS = {
    "embeddings_path": "/home/ovos/NTR-Data/MeePiEmbeddings.npy",
    "memories_data_path": "/home/ovos/NTR-Data/MeePiMemories.json",
    "mee_image_path": "/home/ovos/NTR-Data/cover.jpg",
    "media_folder": "/home/ovos/MeePi_Media",
    "display_mee_image":  True,
    "fallback_friendly": False,  # True to quietly pass unknowns on to AI Brain

    # Tuning parameters (from CONFIG in Python script)
    "top_n": 5,  # Number of top results to return
    "similarity_threshold": 0.35,  # Minimum similarity score to consider a match
    "model_name": "all-MiniLM-L6-v2"  # Embedding model
}


class NearTotalRecall(OVOSSkill):
    def initialize(self):

        self.log.info(f"Initializing Variables ... to be safe")
        self.enabled = False
        self.is_reciting = False  # Track if MeePi is currently babbling

        # These will be populated later
        self.embeddings = None
        self.memory_data = None
        self.model = None
        self.media_available = False

        # Static paths and settings (may not change often)
        self.embeddings_path = None
        self.memories_data_path = None
        self.image_path = None
        self.media_folder = None

        # Runtime tuning settings
        self.top_n = None
        self.similarity_threshold = None
        self.model_name = None
        self.display_mee_image = None
        self.fallback_on = None

        # merge default settings
        # self.settings is a jsondb, which extends the dict class and adds helpers like merge
        self.settings.merge(DEFAULT_SETTINGS, new_only=True)

        self.load_databanks()

        self.log.info(f"Done with Initialize")

    def load_databanks(self):
        self.log.info("Initializing Near-Total-Recall Memory Banks")
        # self.log.info(f"Skill ID: {self.skill_id}")

        # Initial Initialization
        self.enabled = True  # an optimist!

        # Load settings from self.settings
        self.embeddings_path = self.settings.get("embeddings_path")
        self.memories_data_path = self.settings.get("memories_data_path")
        self.image_path = self.settings.get("mee_image_path")
        self.media_folder = self.settings.get("media_folder")

        self.display_mee_image = self.settings.get("display_mee_image")
        self.fallback_on = self.settings.get("fallback_friendly")
        self.top_n = self.settings.get("top_n")
        self.similarity_threshold = self.settings.get("similarity_threshold")
        self.model_name = self.settings.get("model_name")

        # Initialize with paths to the memory_bank and embeddings.

        try:
            self.embeddings = np.load(self.embeddings_path)
            # ✅ Normalize memory embeddings (unit vectors for cosine similarity)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            if np.any(norms == 0):
                self.log.warning("Zero-norm vectors found in memory embeddings. Check your data.")
            self.embeddings = self.embeddings / norms
        except Exception as e:
            self.log.error(f"Failed to load embeddings: {e}")
            self.embeddings = None
            self.enabled = False

        try:
            with open(self.memories_data_path, 'r', encoding='utf-8') as f:
                self.memory_data = json.load(f)
            self.log.info(f"Loaded {len(self.memory_data)} memories from JSON.")
        except Exception as e:
            self.log.error(f"Failed to load memory JSON: {e}")
            self.memory_data = None
            self.enabled = False

        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            self.log.error(f"Failed to load Sentence Transformer model: {e}")
            self.model = None
            self.enabled = False

        if not os.path.isdir(self.media_folder):
            self.log.warning(f"Media folder does not exist: {self.media_folder}")
            self.media_available = False
            self.speak("Memory Palace disabled:  Media Folder does not exist")
        else:
            self.media_available = True

        # Notify the user if something went wrong
        if self.memory_data is None or self.embeddings is None or self.model is None:
            self.speak_dialog("error_initialization")

        self.log.info(f"MeePi Databank Initialization Complete")

    @classproperty
    def runtime_requirements(self):
        return RuntimeRequirements(
            internet_before_load=False,
            network_before_load=False,
            gui_before_load=False,
            requires_internet=False,
            requires_network=False,
            requires_gui=False,
            no_internet_fallback=True,
            no_network_fallback=True,
            no_gui_fallback=True,
        )

    @property
    def my_setting(self):
        """Dynamically get the my_setting from the skill settings file.
        If it doesn't exist, return the default value.
        This will reflect live changes to settings.json files (local or from backend)
        """
        return self.settings.get("my_setting", "default_value")

    def find_closest_memory(self, query):
        """
        This method searches for the most similar memories based on the query using cosine similarity or other methods.
        """
        if self.memory_data is None or self.embeddings is None:
            self.log.error("Cleaned data or Embeddings not loaded.")
            return []

        self.log.info(f"🔍 Finding closest memory for query: '{query}'")

        # OLD Use the model to encode the query
        # OLD query_embedding = self.model.encode([query])

        # NEW Get just the vector for better similarities
        query_embedding = self.model.encode([query])[0]  # Get just the vector
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Compute similarity between query and all memory embeddings
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()

        # Find the top N most similar memories
        top_n_indices = np.argsort(similarities)[::-1][:self.top_n]
        # OLD results = [(similarities[i], self.memory_data[i], self.memory_data[i]['Timestamp']) for i in
        # OLD            top_n_indices]
        results = [(similarities[i], self.memory_data[i], self.memory_data[i]['Timestamp'],
                    self.memory_data[i].get("Title", "")) for i in top_n_indices]

        self.log.info("📊 Top memory matches:")
        for rank, (score, memory, memory_id, memory_title) in enumerate(results, start=1):
            self.log.info(f"  {rank}. Title: '{memory_title}' | Score: {score:.4f}")

        # If top match is below threshold, pretend nothing was found
        if results and results[0][0] < self.similarity_threshold:
            return []

        return results

    def display_cover_image(self, memory):
        """If a cover.jpg exists for the memory, show it on the GUI."""
        raw_timestamp = memory.get("Timestamp", "")
        title = memory.get("Title", "").lower().replace(" ", "_")

        # Convert human-readable timestamp to sortable format
        try:
            dt_obj = datetime.strptime(raw_timestamp, "%m/%d/%Y %H:%M:%S")
            sortable_ts = dt_obj.strftime("%Y%m%d%H%M%S")
        except Exception as e:
            self.log.warning(f"Could not parse timestamp '{raw_timestamp}': {e}")
            sortable_ts = "unknown_time"

        # Build the expected folder path
        folder_name = f"{sortable_ts}_{title}"
        cover_path = os.path.join(self.media_folder, folder_name, "cover.jpg")

        self.log.info(f"Looking for cover.jpg at: {cover_path}")
        if os.path.exists(cover_path):
            self.gui.show_image(cover_path, fill='PreserveAspectFit')
            self.log.info("✅ Found cover.jpg — displaying it.")
        else:
            self.log.info("❌ No cover.jpg found for this memory.")

    def recall_full_memory(self, memory_id):
        """
        This method retrieves the full memory details (e.g., from MeePiMemories.csv) using the memory ID.
        We'll also see if it is long-winded and offer a summary
        """
        if self.memory_data is None:
            self.log.error("Original data not loaded.")
            return None

        # Assuming memory_id corresponds to the 'Timestamp' or another unique field
        # Find the memory dictionary with the matching timestamp
        memory_row = [m for m in self.memory_data if m['Timestamp'] == memory_id]

        if not memory_row:
            return None  # No match found

        # We CAN remember!
        memory = memory_row[0]

        # Project image of MeeSelf (if option set)
        if self.display_mee_image:
            self.gui.show_image(self.image_path, fill='PreserveAspectFit')

        # Extract details
        description = memory['Memory_Description']
        is_long = memory.get("is_long_story", False)
        has_summary = bool(memory.get("Memory_Summary"))

        # Warn the user and offer summary if available
        if is_long:
            response = self.get_response("long_story_warning")  # Ask user for choice
            if response and "summary" in response.lower() and has_summary:
                return memory["Memory_Summary"]  # Return summary

        return description  # Default to full memory

    @intent_handler("DoYouRecall.intent")
    def handle_do_you_recall_intent(self, message):
        if not self.enabled:
            self.speak_dialog("ntr_disabled_due_to_error")
            return

        query = message.data.get("query", "")
        self.log.info(f"Received query for recall: {query}")

        # Check for exact title match (case_insensitive)
        exact_match = [m for m in self.memory_data if m['Title'].lower() == query.lower()]
        if exact_match:
            self.log.info(f"Found exact match for query: {query}")
            memory_dict = exact_match[0]  # <== ADDED: give it a name for clarity
            memory_content = self.recall_full_memory(exact_match[0]['Timestamp'])
            if memory_content:
                dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
                self.is_reciting = True
                if self.media_available:
                    self.display_cover_image(memory_dict)  # <== FIXED: pass full dict
                self.speak_dialog(dialog_file, {"memory": memory_content}, wait=True)
                self.is_reciting = False
                return True  # Fallback Friendly 3

        # Fall-thru to find the closest match logic
        self.log.info(f"Finding closest memory for query: {query}")
        results = self.find_closest_memory(query)

        # Handle results
        if results:
            self.log.info(f"RESULTS! For query: {query}")
            similarity, memory_dict, memory_id, memory_title = results[0]  # <== CLARIFIED/FIXED
            self.log.info(f"🧠 Best match: '{memory_title}' (Score: {similarity:.4f})")

            # memory = results[0]  # Take the first match
            memory_content = self.recall_full_memory(memory_id)  # Use timestamp or similar for recall

            if memory_content:
                dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
                self.is_reciting = True
                if self.media_available:  # <== ADDED check
                    self.display_cover_image(memory_dict)  # <== FIXED: pass full dict
                self.speak_dialog(dialog_file, {"memory": memory_content}, wait=True)
                self.is_reciting = False
                return True  # Fallback Friendly
            else:
                self.log.info(f"RESULTS but NO CONTENT For query: {query}")
                if self.fallback_on:
                    return False  # quietly pass on this one Fallback Friendly
                else:
                    self.speak_dialog("no_memory_found")
        else:
            self.log.info(f"NO RESULTS! For query: {query}")
            if self.fallback_on:
                return False  # quietly pass on this one Fallback Friendly
            else:
                self.speak_dialog("no_memory_found")

    @intent_handler("ThanksCatcher.intent")
    def handle_gratitude_poltergeist(self, message):
        self.log.info("🩹 Caught stray 'thank you' — likely OVOS bug.")
        # THIS IS A KLUDGE ... catch strays
        # MeePI Says NOTHING, Knows NOTHING, Does NOTHING
        # Just don't EVER thank MeePI

    def stop(self):
        """ Action to take when "stop" is requested by the user.
        """
        if self.is_reciting:
            self.speak("")  # Stop MeePi from talking
            self.is_reciting = False
            self.speak_dialog("stopped_talking.dialog")  # Feedback
            self.log.info("MeePi was interrupted by user.")
            return True  # Indicate that MeePi stopped
        return False  # Nothing was interrupted
