from ovos_utils import classproperty
from ovos_utils.process_utils import RuntimeRequirements
from ovos_workshop.decorators import intent_handler
from ovos_workshop.skills import OVOSSkill

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer


# NTR data and tuning parameters in <NTR_Skill>/settings.json
DEFAULT_SETTINGS = {
    "embeddings_path": "/home/ovos/NTR-Data/MeePiEmbeddings-0511.npy",
    "memories_data_path": "/home/ovos/NTR-Data/MeePiMemories-0511.json",
    "image_path": "/home/ovos/NTR-Data/cover.jpg",
    "media_folder": "/home/ovos/MeePi_Media",
    "display_mee_image":  True,
    "fallback_friendly": False,  # True to quietly pass unknowns on to AI Brain

    # Tuning parameters (from CONFIG in Python script)
    "top_n": 5,  # Number of top results to return
    "similarity_threshold": 0.5,  # Minimum similarity score to consider a match
    "model_name": "all-MiniLM-L6-v2"  # Embedding model
}


class NearTotalRecall(OVOSSkill):
    def __init__(self, *args, bus=None, **kwargs):
        """The __init__ method is called when the Skill is first constructed.
        Note that self.bus, self.skill_id, self.settings, and
        other base class settings are only available after the call to super().

        This is a good place to load and pre-process any data needed by your
        Skill, ideally after the super() call.
        """
        super().__init__(*args, bus=bus, **kwargs)
        self.learning = True

        self.log.info(f"In __init__:  Initializing Variables ... super() just called")
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

        self.log.info(f"Done with __init__")

    def load_databanks(self):
        self.log.info("Initializing Near-Total-Recall Memory Banks")
        self.log.info(f"Skill ID: {self.skill_id}")

        # Initial Initialization
        self.enabled = True  # an optimist!

        # Load settings from self.settings
        self.embeddings_path = self.settings.get("embeddings_path")
        self.memories_data_path = self.settings.get("memories_data_path")
        self.image_path = self.settings.get("image_path")
        self.media_folder = self.settings.get("media_folder")

        self.display_mee_image = self.settings.get("display_mee_image")
        self.fallback_on = self.settings.get("fallback_friendly")
        self.top_n = self.settings.get("top_n")
        self.similarity_threshold = self.settings.get("similarity_threshold")
        self.model_name = self.settings.get("model_name")

        # Initialize with paths to the memory_bank and embeddings.

        try:
            self.embeddings = np.load(self.embeddings_path)
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
            self.log.error(f"Media folder does not exist: {self.media_folder}")
            self.media_available = False
            self.speak("Visuals disabled:  Media Folder does not exist")
        else:
            self.media_available = True

        # Notify the user if something went wrong
        if self.memory_data is None or self.embeddings is None or self.model is None:
            self.speak_dialog("error_initialization")

        self.log.info(f"Initialization Complete")

    def initialize(self):
        # merge default settings
        # self.settings is a jsondb, which extends the dict class and adds helpers like merge
        self.settings.merge(DEFAULT_SETTINGS, new_only=True)

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

        # Use the model to encode the query
        query_embedding = self.model.encode([query])

        # Compute similarity between query and all memory embeddings
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()

        # Find the top N most similar memories
        top_n_indices = np.argsort(similarities)[::-1][:self.top_n]
        results = [(similarities[i], self.memory_data[i], self.memory_data[i]['Timestamp']) for i in
                   top_n_indices]

        # If top match is below threshold, pretend nothing was found
        if results and results[0][0] < self.similarity_threshold:
            return []

        return results

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

        # Extract details
        description = memory_row[0]['Memory_Description']
        is_long = memory_row[0].get("is_long_story", False) if not memory_row else False
        has_summary = "Memory_Summary" in memory_row

        # Looks Like we will speak - display MeePi image
        if self.display_mee_image:
            self.gui.show_image(self.image_path, fill='PreserveAspectFit')

        # Warn the user and offer summary if available
        if is_long:
            response = self.get_response("long_story_warning")  # Ask user for choice
            if response and "summary" in response.lower() and has_summary:
                return memory_row[0]["Memory_Summary"]  # Return summary

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
            memory_content = self.recall_full_memory(exact_match[0]['Timestamp'])
            if memory_content:
                dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
                self.is_reciting = True
                self.speak_dialog(dialog_file, {"memory": memory_content}, wait=True)
                self.is_reciting = False
                return True  # Fallback Friendly 3

        # Fall-thru to find the closest match logic
        self.log.info(f"Finding closest memory for query: {query}")
        results = self.find_closest_memory(query)

        # Handle results
        if results:
            self.log.info(f"RESULTS! For query: {query}")
            memory = results[0]  # Take the first match
            memory_content = self.recall_full_memory(memory[2])  # Use timestamp or similar for recall

            if memory_content:
                dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
                self.is_reciting = True
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
