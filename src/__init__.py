from ovos_utils import classproperty
from ovos_utils.process_utils import RuntimeRequirements
from ovos_workshop.decorators import intent_handler
from ovos_workshop.skills import OVOSSkill

import os
import re
import json
import random
import time
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter

# NTR data and tuning parameters in <NTR_Skill>/settings.json
DEFAULT_SETTINGS = {
    "embeddings_path": "/home/ovos/NTR-Data/MeePiEmbeddings.npy",
    "memories_data_path": "/home/ovos/NTR-Data/MeePiMemoryBank.json",
    "mee_image_path": "/home/ovos/NTR-Data/cover.jpg",
    "media_folder": "/home/ovos/MeePi_Media",
    "display_mee_image":  True,
    "fallback_friendly": False,  # True to quietly pass unknowns on to AI Brain

    # Tuning parameters
    "top_n": 5,  # Number of top results to return
    "similarity_threshold": 0.32,  # Minimum similarity score to consider a match
    "model_name": "all-MiniLM-L6-v2",  # Embedding model
    "chunk_pause_seconds": 0.4  # Paragraph chunking pause (seconds)
}


class NearTotalRecall(OVOSSkill):
    def __init__(self, *args, **kwargs):
        """The __init__ method is called when the Skill is first constructed.
        Note that self.bus, self.skill_id, self.settings, and
        other base class settings are only available after the call to super().
        """
        super().__init__(*args, **kwargs)
        self.learning = True
        self.is_reciting = False

    def initialize(self):

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
        query_embedding = self.model.encode([query])

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
        """If a cover image exists for the memory, show it on the GUI."""
        raw_timestamp = memory.get("Timestamp", "")
        title = memory.get("Title", "").lower().replace(" ", "_")

        # Convert human-readable timestamp to sortable format
        try:
            dt_obj = datetime.strptime(raw_timestamp, "%m/%d/%Y %H:%M:%S")
            sortable_ts = dt_obj.strftime("%Y%m%d%H%M%S")
        except Exception as e:
            self.log.warning(f"Could not parse timestamp '{raw_timestamp}': {e}")
            sortable_ts = "unknown_time"

        folder_name = f"{sortable_ts}_{title}"
        folder_path = os.path.join(self.media_folder, folder_name)

        # Look for the first valid cover image
        for ext in [".jpg", ".jpeg", ".png"]:
            cover_path = os.path.join(folder_path, f"cover{ext}")
            if os.path.exists(cover_path):
                self.gui.show_image(cover_path, fill='PreserveAspectFit')
                self.log.info(f"✅ Found cover image ({cover_path}) — displaying it.")
                return

        self.log.info("❌ No cover image (jpg/jpeg/png) found for this memory.")

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

    def speak_buffered(self, dialog_file: str, text: str):
        """Speak text in paragraph-sized chunks (delimited by blank lines)."""
        if not text:
            return

        # Load pause from settings
        pause = self.settings.get("chunk_pause_seconds", 0.4)

        # Normalize line breaks
        normalized = text.replace("\r\n", "\n").strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]

        # First paragraph uses the main dialog
        self.speak_dialog(dialog_file, {"memory": paragraphs[0]}, wait=True)

        # Remaining paragraphs
        for p in paragraphs[1:]:
            if not self.is_reciting:
                break
            # self.wait_while_speaking()
            time.sleep(pause)
            # Use your optional chunk dialog, otherwise raw speak
            self.speak_dialog("recite_chunk", {"memory": p}, wait=True)

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
                # self.speak_dialog(dialog_file, {"memory": memory_content}, wait=True)
                self.speak_buffered(dialog_file, memory_content)
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
                # self.speak_dialog(dialog_file, {"memory": memory_content}, wait=True)
                self.speak_buffered(dialog_file, memory_content)
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

    @intent_handler("MemoryChecker.intent")
    def handle_memory_checker_intent(self, _message):
        self.speak("Checking MeePi Memory Banks")
        total_memories = len(self.memory_data)
        era_counts = Counter(m.get("Era", "Unknown") or "Unknown" for m in self.memory_data)

        # Build formatted era list
        era_list = []
        for era, count in era_counts.items():
            if era.lower() == "unknown":
                era_list.append(f"{count} with no era assigned")
            else:
                era_list.append(f"{count} from {era}")

        # Combine into natural-sounding string
        if len(era_list) > 1:
            era_summary = ", ".join(era_list[:-1]) + f", and {era_list[-1]}"
        else:
            era_summary = era_list[0]

        self.speak(f"I have {total_memories} memories: {era_summary}.")
        return

    @intent_handler("RandomMemory.intent")
    def handle_random_memory_intent(self, _message):
        if not self.enabled:
            self.speak_dialog("ntr_disabled_due_to_error")
            return

        # Pick a random memory
        memory = random.choice(self.memory_data)
        memory_id = memory["Timestamp"]

        # memory = results[0]  # Take the first match
        memory_content = self.recall_full_memory(memory_id)  # Use timestamp or similar for recall

        if memory_content:
            self.speak_dialog("random_memory", {"era": memory_content})
            dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
            self.is_reciting = True
            if self.media_available:  # <== ADDED check
                self.display_cover_image(memory)  # <== FIXED: pass full dict
            # self.speak_dialog(dialog_file, {"memory": memory_content}, wait=True)
            self.speak_buffered(dialog_file, memory_content)
            self.is_reciting = False
            return True  # Fallback Friendly
        else:
            self.log.info(f"RESULTS but NO CONTENT For Random Memory")
            if self.fallback_on:
                return False  # quietly pass on this one Fallback Friendly
            else:
                self.speak_dialog("no_memory_found")
        return

    @intent_handler("ThanksCatcher.intent")
    def handle_gratitude_poltergeist(self, _message):
        self.log.info("🩹 Caught stray 'thank you' — likely OVOS bug.")
        # THIS IS A KLUDGE ... catch strays
        # MeePI Says NOTHING, Knows NOTHING, Does NOTHING
        # Just don't EVER thank MeePI

    def stop(self):
        """ Action to take when "stop" is requested by the user.
        """
        if self.is_reciting:
            self.is_reciting = False
            self.speak_dialog("stopped_talking")  # Feedback
            self.log.info("MeePi was interrupted by user.")
            return True  # Indicate that MeePi stopped
        return False  # Nothing was interrupted
