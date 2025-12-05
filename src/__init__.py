# Copyright 2024 TPed
#
# Licensed under the MIT License.
# You may obtain a copy of the License at https://opensource.org/licenses/MIT

from ovos_utils import classproperty
from ovos_utils.process_utils import RuntimeRequirements
from ovos_workshop.decorators import intent_handler
from ovos_workshop.skills import OVOSSkill
from ovos_bus_client.session import SessionManager
from ovos_bus_client.message import Message

import os
import re
import json
import random
import time
import math
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter

# NTR data and tuning parameters in <NTR_Skill>/settings.json
DEFAULT_SETTINGS = {
    "embeddings_path": "/home/ovos/NTR-Data/MeePiEmbeddings.npy",
    "memories_data_path": "/home/ovos/NTR-Data/MeePiMemoryBank.json",
    "mee_image_path": "/home/ovos/NTR-Data/cover.jpg",
    "media_folder": "/home/ovos/MeePi_MemoryPalace",
    "display_mee_image":  True,
    "fallback_friendly": False,  # True to quietly pass unknowns on to AI Brain

    # Tuning parameters
    "top_n": 5,  # Number of top results to return
    "similarity_threshold": 0.32,  # Minimum similarity score to consider a match
    "model_name": "all-MiniLM-L6-v2",  # Embedding model
    "chunk_pause_seconds": 0.3,  # Paragraph chunking pause (seconds)
    "max_tts_chunk_size": 600   # Max Characters per TTS call
}


class NearTotalRecall(OVOSSkill):
    def __init__(self, *args, **kwargs):
        """The __init__ method is called when the Skill is first constructed.
        Note that self.bus, self.skill_id, self.settings, and
        other base class settings are only available after the call to super().
        """
        super().__init__(*args, **kwargs)
        self.session_results = {}
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
        self.chunk_pause = self.settings.get("chunk_pause_seconds")
        self.max_chunk_size = self.settings.get("max_tts_chunk_size", 250)

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
        self.speak("MeePi Near Total Recall is Alive.  Version 0 dot 9.  Hand-off to Visual Recall")

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
        raw_title = memory.get("Title", "")

        # Convert human-readable timestamp to sortable format
        try:
            dt_obj = datetime.strptime(raw_timestamp, "%m/%d/%Y %H:%M:%S")
            sortable_ts = dt_obj.strftime("%Y%m%d%H%M%S")
        except Exception as e:
            self.log.warning(f"Could not parse timestamp '{raw_timestamp}': {e}")
            sortable_ts = "unknown_time"

        # Match MeePi_MediaFoldersV6.py sanitation
        safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_title).lower()
        folder_name = f"{sortable_ts}_{safe_title}"
        folder_path = os.path.join(self.media_folder, folder_name)

        # Look for the first valid cover image
        for ext in [".jpg", ".jpeg", ".png"]:
            cover_path = os.path.join(folder_path, f"cover{ext}")
            if os.path.exists(cover_path):
                # Use memory's TTS time, fallback to 20 if missing
                tts_time = memory.get("tts_time_seconds", 20)
                hold_time = math.ceil(tts_time + 1)  # round UP, add 1s cushion
                self.gui.show_image(
                    cover_path,
                    fill="PreserveAspectFit",
                    override_idle=hold_time
                )
                self.log.info(f"✅ Found cover image ({cover_path}) — displaying it.")
                return

        self.log.info("❌ No cover image (jpg/jpeg/png) found for this memory.")

    def recall_full_memory(self, memory_id):
        """
        This method retrieves the full memory details (e.g., from MeePiMemoryBanks) using the memory ID.
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

        # Use memory's TTS time, fallback to 20 if missing
        tts_time = memory.get("tts_time_seconds", 20)
        hold_time = math.ceil(tts_time + 1)  # round UP, add 1s cushion

        # Project image of MeeSelf (if option set)
        if self.display_mee_image:
            self.gui.show_image(self.image_path, fill='PreserveAspectFit', override_idle=hold_time)

        # Extract details
        description = memory['Memory_Description']
        is_long = memory.get("is_long_story", False)
        has_summary = bool(memory.get("Memory_Summary"))

        # Warn the user and offer summary if available
        if is_long:
            response = self.get_response("long_story_warning")  # Ask user for choice
            if "full" in response.lower() or "fall" in response.lower() or "all" in response.lower():
                return description  # Explicit full request
            elif response and "summary" in response.lower() and has_summary:
                return memory["Memory_Summary"]  # Return summary
            elif has_summary:
                return memory["Memory_Summary"]  # Return summary
        # Default if user gives no usable response - should be short
        return description  # Default to full memory

    def send_visual_recall_request(self, memory_dict):
        """
        Calculates folder path, counts all media types, asks user for confirmation,
        and sends Messagebus request to Visual Recall skill if confirmed.
        """
        if not self.media_available:
            self.log.info("Visual recall skipped: Media folder not available.")
            self.gui.release()  # GUI Release
            return

        raw_timestamp = memory_dict.get("Timestamp", "")
        raw_title = memory_dict.get("Title", "")

        # Reconstruct the exact folder path (same logic as before)
        try:
            dt_obj = datetime.strptime(raw_timestamp, "%m/%d/%Y %H:%M:%S")
            sortable_ts = dt_obj.strftime("%Y%m%d%H%M%S")
        except ValueError:
            self.log.warning(f"Could not parse timestamp '{raw_timestamp}'.")
            sortable_ts = "unknown_time"
        except TypeError:
            self.log.warning(f"Timestamp was unexpected type.")
            sortable_ts = "unknown_time"

        # Match MeePi_MediaFoldersV6.py sanitation
        safe_title = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_title).lower()
        folder_name = f"{sortable_ts}_{safe_title}"
        folder_path = os.path.join(self.media_folder, folder_name)

        if not os.path.isdir(folder_path):
            self.log.warning(f"Visual recall skipped: Folder not found at {folder_path}")
            self.speak_dialog("end_of_memory")
            self.gui.release()  # <-- CORRECTED: GUI Release
            return

        # --- 1. COUNT AND QUANTIFY ALL MEDIA ---

        # Supported media extensions across all types (image, audio, video)
        supported_ext = (
            '.jpg', '.jpeg', '.png', '.gif',  # Images
            '.mp4', '.mkv', '.avi', '.mov', '.webm',  # Videos
            '.mp3', '.wav', '.flac', '.m4a', '.aac'  # Audio
        )

        all_media_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_ext)]

        # Filter out the cover image (it's handled separately by NTR/VR initial display)
        files_to_display = [
            f for f in all_media_files
            if os.path.splitext(os.path.basename(f).lower())[0] != "cover"
        ]

        media_count = len(files_to_display)

        # --- 2. BRANCHING LOGIC ---

        if media_count == 0:
            # Scenario: Only a cover image or no media to display
            self.log.info("Visual recall: Only cover image or no media found.")
            # Dialog: "That's all I remember."
            self.speak_dialog("end_of_memory")
            self.gui.release()  # GUI Release
            return

        # Determine the quantifier for the user dialog
        if media_count == 1:
            quantifier_text = "one related image"
        elif media_count <= 4:
            quantifier_text = "a few images"
        else:
            quantifier_text = "a lot of images"

        # --- 3. ASK FOR USER CONFIRMATION ---
        self.log.info("Asking user for visual confirmation with quantifier variable.")

        # Pass the quantifier text to the visual_media_prompt.dialog file
        response = self.get_response(
            "visual_media_prompt",
            data={"quantifier": quantifier_text},
            num_retries=1
        )

        # --- 4. SEND SIGNAL OR FINISH ---

        if response and self.voc_match(response, "yes"):
            self.log.info(f"User confirmed: Sending visual recall request for {media_count} items.")
            self.bus.emit(
                Message(
                    "visual.recall.display",  # The event registered in the VR skill
                    data={
                        "media_path": folder_path,
                        "title": raw_title
                    }
                )
            )
        else:
            self.log.info("User declined visual recall.")
            self.speak_dialog("visual_declined")
            self.gui.release()
            return

    @staticmethod
    def _smart_chunk(text, limit):
        """
        Splits text into chunks of roughly 'limit' characters,
        but avoids breaking sentences in the middle.
        """
        # Split by sentence endings (. ? ! followed by whitespace)
        sentences = re.split(r'(?<=[.?!])\s+', text)

        chunks = []
        current_chunk = ""

        for s in sentences:
            # If adding this sentence stays under the limit, add it
            if len(current_chunk) + len(s) < limit:
                current_chunk += s + " "
            else:
                # Limit reached: push current chunk and start a new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = s + " "

        # Don't forget the leftovers
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def speak_buffered(self, dialog_file: str, text: str):
        if not text:
            return

        pause = self.chunk_pause
        self.is_reciting = True

        # Normalize line breaks to handle paragraphs
        normalized = text.replace("\r\n", "\n").strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]

        try:
            for i, p in enumerate(paragraphs):
                if not self.is_reciting:
                    break

                # --- LOGIC START ---
                text_to_chunk = p

                # 1. OPTIMIZATION: If it's the very first paragraph,
                # peel off the first sentence for immediate playback.
                if i == 0 and len(p) > 60:
                    match = re.search(r'(?<=[.?!])\s+', p)
                    if match:
                        first_sentence = p[:match.start()]
                        text_to_chunk = p[match.end():]  # The rest gets chunked below

                        # SPEAK FIRST SENTENCE IMMEDIATELY
                        self.speak_dialog("recite_chunk", {"memory": first_sentence}, wait=True)

                # 2. Smart Chunk the remaining text
                # This prevents the "Loading..." gap between sentence 1 and 2
                chunks = self._smart_chunk(text_to_chunk, self.max_chunk_size)

                for j, chunk in enumerate(chunks):
                    if not self.is_reciting:
                        break

                    # Use generic 'recite_chunk' unless it's the very first block of the entire memory
                    # (This handles your original dialog_file logic)
                    dialog = dialog_file if (i == 0 and j == 0 and text_to_chunk == p) else "recite_chunk"

                    self.speak_dialog(dialog, {"memory": chunk}, wait=True)

                # Paragraph Pause (only if there is more coming)
                if i < len(paragraphs) - 1:
                    time.sleep(pause)

        finally:
            self.is_reciting = False

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
                # speak memory title immediately after
                title = memory_dict.get("Title", "this one")
                self.speak(f"I clearly remember {title}!", wait=False)
                dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
                if self.media_available:
                    self.display_cover_image(memory_dict)  # <== FIXED: pass full dict
                # self.speak_dialog(dialog_file, {"memory": memory_content}, wait=True)
                self.speak_buffered(dialog_file, memory_content)
                # NEW: Hand off to Visual Recall (handles GUI release internally)
                self.send_visual_recall_request(memory_dict)
                return True  # Fallback Friendly 3

        # Fall-thru to find the closest match logic
        self.log.info(f"Finding closest memory for query: {query}")
        results = self.find_closest_memory(query)

        # Handle results
        if results:
            self.log.info(f"RESULTS! For query: {query}")
            similarity, memory_dict, memory_id, memory_title = results[0]  # <== CLARIFIED/FIXED
            self.log.info(f"🧠 Best match: '{memory_title}' (Score: {similarity:.4f})")

            # --- Tune-up Step 1: announce memory found (non-blocking) ---
            era_name = memory_dict.get("Era", "the past") or "the past"
            mem_type = memory_dict.get("Memory_Type", "personal") or "personal"

            # speak preamble from memory_found.dialog
            self.speak_dialog(
                "memory_found",
                {"era_name": era_name, "mem_type": mem_type},
                wait=False
            )

            # speak memory title immediately after
            title = memory_dict.get("Title", "this one")
            self.speak(f"It's titled: {title}", wait=False)

            # memory = results[0]  # Take the first match
            memory_content = self.recall_full_memory(memory_id)  # Use timestamp or similar for recall

            if memory_content:
                dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
                if self.media_available:  # <== ADDED check
                    self.display_cover_image(memory_dict)  # <== FIXED: pass full dict
                self.speak_buffered(dialog_file, memory_content)
                # Release GUI only when done with intent
                # NEW: Hand off to Visual Recall (handles GUI release internally)
                self.send_visual_recall_request(memory_dict)
                # self.gui.release() - send_visual_recall will do this
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

        # Grab the era directly from the memory dict
        era_name = memory.get("Era", "past")

        # memory = results[0]  # Take the first match
        memory_content = self.recall_full_memory(memory_id)  # Use timestamp or similar for recall

        if memory_content:
            self.speak_dialog("random_memory", {"era": era_name})
            dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
            if self.media_available:  # <== ADDED check
                self.display_cover_image(memory)  # <== FIXED: pass full dict
            # self.speak_dialog(dialog_file, {"memory": memory_content}, wait=True)
            self.speak_buffered(dialog_file, memory_content)
            # NEW: Hand off to Visual Recall
            self.send_visual_recall_request(memory)
            # self.gui.release() - send_visual will take care of this
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
        session = SessionManager.get()
        # called during global stop only

        if session.session_id in self.session_results:
            self.session_results.pop(session.session_id)
        if session.session_id == "default":
            self.gui.release()

        if self.is_reciting:
            self.is_reciting = False
            self.speak_dialog("stopped_talking")  # Feedback
            self.log.info("MeePi was interrupted by user.")
