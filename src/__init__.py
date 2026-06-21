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
import filecmp
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from .version import (
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_BUILD,
    VERSION_ALPHA
)

# NTR data and tuning parameters in <NTR_Skill>/settings.json
DEFAULT_SETTINGS = {
    "birth_date": "1958-05-20",  # Set to your birthdate!
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
    "chunk_pause_seconds": 0.2,  # Paragraph chunking pause (seconds)
    "max_tts_chunk_size": 300,  # Max Characters per TTS call

    # Hybrid search weights (must sum to 1.0)
    # Semantic: embedding cosine similarity (good for concepts, context)
    # Keyword:  rapidfuzz + exact token matching (good for names, places)
    "semantic_weight": 0.60,
    "keyword_weight":  0.40
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

        """ # Placeholders to stop the warnings
        self.log_level = "INFO"
        self.enabled = False
        self.embeddings_path = ""
        self.memories_data_path = ""
        self.image_path = ""
        self.media_folder = ""
        self.display_mee_image = True
        self.fallback_on = False
        self.top_n: int = 3                         # PyCharm: "This is always a number"
        self.similarity_threshold: float = 0.5      # PyCharm: "This is always a number"
        self.model_name = ""
        self.chunk_pause: float = 0.2               # PyCharm: "This is always a number"
        self.max_chunk_size = 250
        self.embeddings: np.ndarray | None = None   # PyCharm: "This is a numpy array"
        self.memory_data: list[dict] = []           # PyCharm: "Items inside are dictionaries"
        self.model: SentenceTransformer | None = None
        self.media_available = False
        """

    def initialize(self):
        # merge default settings
        # self.settings is a jsondb, which extends the dict class and adds helpers like merge
        self.settings.merge(DEFAULT_SETTINGS, new_only=True)
        self.log_level = self.settings.get("log_level", "INFO")

        self.load_databanks()

        # Speak version if log_level != INFO
        if self.log_level.upper() != "INFO":
            ver = self.skill_version()
            spoken_version = ver.replace("a", " alpha ")
            self.speak(
                f"MeePi Near Total Recall, version {spoken_version}, initialized",
                wait=False
            )
            # Safety check - report gui state
            if self.gui:
                # self.gui_mode = True
                self.speak("GUI detected and enabled.")
            else:
                # self.gui_mode = False
                self.speak("self.gui is NOT set - GUI MODE forced on")

        self.log.info(f"Done with Initialize")

    def load_databanks(self):
        self.log.info("Initializing Near-Total-Recall Memory Banks")
        # self.log.info(f"Skill ID: {self.skill_id}")

        # Initial Initialization
        self.enabled = True  # an optimist!

        # Load settings from self.settings
        self.embeddings_path = self.settings.get("embeddings_path") or ""
        self.memories_data_path = self.settings.get("memories_data_path") or ""
        self.image_path = self.settings.get("mee_image_path") or ""
        self.media_folder = self.settings.get("media_folder") or ""

        self.display_mee_image = self.settings.get("display_mee_image", True)
        self.fallback_on = self.settings.get("fallback_friendly", False)
        self.top_n = self.settings.get("top_n", 3)
        self.similarity_threshold = self.settings.get("similarity_threshold", 0.5)
        self.model_name = self.settings.get("model_name") or ""
        self.chunk_pause = self.settings.get("chunk_pause_seconds", 0.2)
        self.max_chunk_size = self.settings.get("max_tts_chunk_size", 250)
        self.semantic_weight = self.settings.get("semantic_weight", 0.60)
        self.keyword_weight = self.settings.get("keyword_weight", 0.40)

        # Initialize with paths to the memory_bank and embeddings.

        try:
            self.embeddings = np.load(self.embeddings_path)
        except Exception as e:
            self.log.error(f"Failed to load embeddings: {e}")
            self.embeddings = None
            self.enabled = False
            self.speak("NTR disabled:  Could not load Embeddings")

        try:
            with open(self.memories_data_path, 'r', encoding='utf-8') as f:
                self.memory_data = json.load(f)
            self.log.info(f"Loaded {len(self.memory_data)} memories from JSON.")
        except Exception as e:
            self.log.error(f"Failed to load memory JSON: {e}")
            self.memory_data = []
            self.enabled = False
            self.speak("NTR disabled:  Could not load Memory Banks")

        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            self.log.error(f"Failed to load Sentence Transformer model: {e}")
            self.model = None
            self.enabled = False
            self.speak("NTR disabled:  Could not load Sentence Transformer Model")

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

    @staticmethod
    def skill_version():
        version_string = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
        if VERSION_ALPHA and int(VERSION_ALPHA) > 0:
            version_string += f"a{VERSION_ALPHA}"
        return version_string

    @property
    def my_setting(self):
        """Dynamically get the my_setting from the skill settings file.
        If it doesn't exist, return the default value.
        This will reflect live changes to settings.json files (local or from backend)
        """
        return self.settings.get("my_setting", "default_value")

    @staticmethod
    def _flip_pronouns(text):
        """Flip third-person or user-centric pronouns to first-person for MeePi."""
        if not text:
            return ""
        # Pronoun mapping for the persona flip
        swaps = {
            r"\byour\b": "my",  # User says "your wedding" -> MeePi says "my wedding"
            r"\bhis\b": "my",  # Title says "his wedding" -> MeePi says "my wedding"
            r"\bher\b": "my",  # Title says "her wedding" -> MeePi says "my wedding"
            r"\bhe\b": "I",  # Title says "he went" -> MeePi says "I went"
            r"\bshe\b": "I",
            r"\bhim\b": "me"
        }

        result = text
        for pattern, replacement in swaps.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Cleanup whitespace
        return re.sub(r'\s+', ' ', result).strip()

    def _map_user_query_to_era(self, query):
        """Translates natural language queries into MeePi Era categories."""
        query = query.lower()

        # 1. Simple Keyword Mapping
        mapping = {
            "childhood": ["childhood", "kid", "child", "growing up", "little", "elementary"],
            "teenage": ["teenage", "teens", "high school", "adolescent", "junior high"],
            "twenties": ["twenties", "20s", "college", "young adult"],
            "midlife": ["midlife", "middle age", "thirties", "forties", "fifties", "30s", "40s", "50s"],
            "senior": ["senior", "retirement", "sixties", "70s", "80s", "90s"],
            "encore": ["encore", "recent", "lately", "now", "today"]
        }

        for era, synonyms in mapping.items():
            if any(syn in query for syn in synonyms):
                return era

        # 2. Decade & Year Math
        # Matches "1975" or "70s"
        match = re.search(r"(\d{4}|\d{2})s?", query)
        if match:
            year_val = int(match.group(1))
            if year_val < 100:
                year_val += 1900

            birth_str = self.settings.get("birth_date", "1958-05-20")
            try:
                birth_year = int(birth_str.split("-")[0])
            except (ValueError, IndexError):
                birth_year = 1958

            age_at_time = year_val - birth_year

            if age_at_time < 13: return "childhood"
            if age_at_time < 20: return "teenage"
            if age_at_time < 30: return "twenties"
            if age_at_time < 60: return "midlife"
            return "senior"

        return query  # Fallback to exactly what was said

    def converse(self, message=None):
        # 0.  Make sure we have a message
        if message is None:
            return False
        # 1. Get the text
        # Original:  utterances = message.data.get('utterances', [])
        utterances = (message.data or {}).get('utterances', [])
        if not utterances:
            return False
        utt = utterances[0].lower().strip()

        # 2. THE STOP ESCAPE (Always let the user kill the process)
        if self.voc_match(utt, "stop") or "stop" in utt:
            self.log.info("NTR Shield: Stop requested. Killing recital.")
            self.stop()  # This calls your skill's stop() method
            return False  # Return False so the system-wide 'Stop' also fires

        # 3. THE RECITING SHIELD
        # If we are in the middle of a memory, we 'swallow' everything else.
        if self.is_reciting:
            # If it's empty, or "thanks", or random noise...
            # We return True to tell OVOS 'I handled this (by ignoring it)'
            self.log.info(f"NTR Shield: Swallowing noise/filler: '{utt}'")
            return True

            # 4. If we aren't reciting, let the brain work normally
        return False

    @staticmethod
    def prep_for_math(text):
        """Normalize text for keyword scoring: lowercase, expand hyphens, strip punctuation."""
        if not isinstance(text, str): return ""
        text = text.lower()
        text = text.replace("pedersen", "pedersen peterson peederson")
        text = text.replace("-", " ")  # hyphens become spaces BEFORE stripping
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text

    @staticmethod
    def scrub_query(q):
        """Strip intent carrier phrases from the raw query before scoring."""
        q = q.lower()
        stops = ["tell me about", "do you remember", "what is", "recall", "the"]
        for word in stops: q = q.replace(word, "")
        return NearTotalRecall.prep_for_math(q).strip()

    def find_closest_memory(self, query):
        """
        Hybrid search: weighted combination of semantic (cosine similarity)
        and keyword (rapidfuzz + exact token + sidekick bonus) scoring.
        """
        if self.memory_data is None or self.embeddings is None or self.model is None:
            self.log.error("Memory data or embeddings not loaded.")
            return []

        self.log.info(f"🔍 Hybrid search for query: '{query}'")

        clean_q = self.scrub_query(query)
        q_vec = self.model.encode([clean_q])
        q_tokens = clean_q.split()

        # ── Semantic scores ──────────────────────────────────────────────
        sem_scores = cosine_similarity(q_vec, self.embeddings)[0]

        # ── Keyword scores ───────────────────────────────────────────────
        keyword_boost = np.zeros(len(self.memory_data))
        for i, memory in enumerate(self.memory_data):
            title = str(memory.get('Title', '')).lower()
            sidekicks = str(memory.get('Sidekicks', '')).lower()
            candidate = self.prep_for_math(title + " " + sidekicks)

            # Tier 1: fuzzy partial match across title+sidekicks
            ratio = fuzz.partial_ratio(clean_q, candidate) / 100.0

            # Tier 2: exact token hits in title+sidekicks
            exact_hits = sum(1 for tok in q_tokens if tok in candidate)
            exact_bonus = exact_hits * 0.15

            # Tier 3: exact token hits in sidekicks only (hand-curated = stronger signal)
            sidekick_tokens = self.prep_for_math(sidekicks).split()
            sidekick_hits = sum(1 for tok in q_tokens if tok in sidekick_tokens)
            sidekick_bonus = sidekick_hits * 0.35

            keyword_boost[i] = (ratio * 0.3) + exact_bonus + sidekick_bonus

        # ── Combined score ───────────────────────────────────────────────
        combined = (sem_scores * self.semantic_weight) + (keyword_boost * self.keyword_weight)
        top_idx = combined.argsort()[-self.top_n:][::-1]

        results = [
            (combined[i], self.memory_data[i], self.memory_data[i]['Timestamp'],
             self.memory_data[i].get('Title', ''))
            for i in top_idx
        ]

        self.log.info("📊 Top hybrid matches:")
        for rank, (score, memory, memory_id, memory_title) in enumerate(results, start=1):
            self.log.info(f"  {rank}. Title: '{memory_title}' | Score: {score:.4f}")

        # If top match is below threshold, pretend nothing was found
        if results and results[0][0] < self.similarity_threshold:
            return []

        return results

    def display_cover_image(self, memory):
        """Show MeeSelf avatar immediately, then replace with cover image if one exists."""
        raw_timestamp = memory.get("Timestamp", "")
        raw_title = memory.get("Title", "")

        # Estimate hold time from memory word count (roughly 2.5 words/second)
        description = memory.get("Memory_Description", "")
        word_count = len(description.split())
        hold_time = max(20, int(word_count / 2.5) + 5)

        # ALWAYS show MeeSelf first — no gap, no OVOS logo flash
        if self.gui and self.display_mee_image:  # temp, don't trust self.gui
            self.gui.show_image(self.image_path, fill='PreserveAspectFit', override_idle=hold_time)
            self.log.info("👤 MeeSelf avatar displayed.")

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

        # OVERLAY: Replace MeeSelf with cover image if one exists
        for ext in [".jpg", ".jpeg", ".png"]:
            cover_path = os.path.join(folder_path, f"cover{ext}")
            if os.path.exists(cover_path):
                if self.gui:
                    self.gui.show_image(cover_path, fill="PreserveAspectFit", override_idle=hold_time)
                    self.log.info(f"🖼 Cover image overlaid ({cover_path})")
                break

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
        # De-implemented
        #tts_time = memory.get("tts_time_seconds", 20)
        #hold_time = math.ceil(tts_time + 1)  # round UP, add 1s cushion

        # Extract details
        description = memory['Memory_Description']
        is_long = memory.get("is_long_story", False)
        has_summary = bool(memory.get("Memory_Summary"))

        # Warn the user and offer summary if available
        if is_long:
            response = self.get_response("long_story_warning") or "" # Ask user for choice
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
            if self.gui:
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
            if self.gui:
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

        # Identify the cover image path for comparison
        cover_path = None
        for f in all_media_files:
            if os.path.splitext(f)[0].lower().strip() == "cover":
                cover_path = os.path.join(folder_path, f)
                break

        # Build the display list by filtering out the cover and its exact clones
        files_to_display = []
        for f in all_media_files:
            file_path = os.path.join(folder_path, f)
            file_root = os.path.splitext(f)[0].lower().strip()

            # Skip the actual file named 'cover'
            if file_root == "cover":
                continue

            # Skip any file that is a duplicate of the cover (The Plymouth Fix)
            if cover_path and filecmp.cmp(file_path, cover_path, shallow=True):
                self.log.info(f"NTR: Ignoring {f} - it is a duplicate of the cover image.")
                continue

            files_to_display.append(f)

        media_count = len(files_to_display)

        # --- 2. BRANCHING LOGIC ---

        if media_count == 0:
            # Scenario: Only a cover image or no media to display
            self.log.info("Visual recall: Only cover image or no media found.")
            # Dialog: "That's all I remember."
            self.speak_dialog("end_of_memory")
            if self.gui:
                self.gui.release()  # GUI Release
            return

        # Determine the quantifier for the user dialog
        if media_count == 1:
            quantifier_text = "a single related image"
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
            # --- BRIDGE THE GAP ---
            # Re-show the current image to prevent the OVOS logo from flashing
            # while the Visual Recall skill is loading.
            if self.gui:
                image_to_hold = cover_path if cover_path else self.image_path
                self.gui.show_image(image_to_hold, fill='PreserveAspectFit', override_idle=20)

            self.bus.emit(
                Message(
                    "visual.recall.display",  # The event registered in the VR skill
                    data={
                        "media_path": folder_path,
                        "title": raw_title
                    }
                )
            )

            # Wait a moment for VR to "pickup" the screen before we exit
            time.sleep(1.2)

        else:
            self.log.info("User declined visual recall.")
            self.speak_dialog("visual_declined")
            if self.gui:
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

        # pause = self.chunk_pause
        self.is_reciting = True

        # Normalize line breaks to handle paragraphs
        normalized = text.replace("\r\n", "\n").strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]

        try:
            for i, p in enumerate(paragraphs):
                if not self.is_reciting:
                    break

                # HEARTBEAT: Keep skill active so Shield stays up
                self.activate()

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
                        self.speak_dialog(dialog_file, {"memory": first_sentence}, wait=False)

                # 2. Smart Chunk the remaining text
                # This prevents the "Loading..." gap between sentence 1 and 2
                chunks = self._smart_chunk(text_to_chunk, self.max_chunk_size)

                for j, chunk in enumerate(chunks):
                    if not self.is_reciting:
                        break

                    # Determine if we should wait.
                    # We only wait on the VERY LAST chunk of a paragraph
                    # to respect your 'chunk_pause' (paragraph break).
                    # is_last_chunk = (i == len(paragraphs) - 1) and (j == len(chunks) - 1)
                    is_end_of_paragraph = (j == len(chunks) - 1)

                    # Speak the chunk.
                    # By setting wait=False for intermediate chunks, OVOS queues them.
                    self.speak_dialog("recite_chunk", {"memory": chunk}, wait=is_end_of_paragraph)

                # Paragraph Pause - only if not the last paragraph
                if i < len(paragraphs) - 1:
                    time.sleep(self.chunk_pause)

        finally:
            self.is_reciting = False

    @intent_handler("DoYouRecall.intent")
    def handle_do_you_recall_intent(self, message):
        if not self.enabled:
            self.speak_dialog("ntr_disabled_due_to_error")
            return False

        # Show MeeSelf immediately so user knows intent was received
        if self.gui and self.display_mee_image:
            self.gui.show_image(self.image_path, fill='PreserveAspectFit', override_idle=60)

        query = message.data.get("query", "")
        self.log.info(f"Received query for recall: {query}")

        # Check for exact title match (case_insensitive)
        exact_match = [m for m in self.memory_data if m['Title'].lower() == query.lower()]
        if exact_match:
            self.log.info(f"Found exact match for query: {query}")
            memory_dict = exact_match[0]  # <== ADDED: give it a name for clarity
            memory_content = self.recall_full_memory(exact_match[0]['Timestamp']) or ""
            if memory_content:
                # speak memory title immediately after
                # title = memory_dict.get("Title", "this one")
                # self.speak(f"I clearly remember {title}!", wait=False)
                dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
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
            # CHANGED: wait=True prevents her from starting the recital before finishing this intro
            self.speak_dialog(
                "memory_found",
                {"era_name": era_name, "mem_type": mem_type},
                wait=True
            )

            # memory = results[0]  # Take the first match
            memory_content = self.recall_full_memory(memory_id) or ""  # Use timestamp or similar for recall

            if memory_content:
                dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
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
                    return True
        else:
            self.log.info(f"NO RESULTS! For query: {query}")
            if self.fallback_on:
                return False  # quietly pass on this one Fallback Friendly
            else:
                self.speak_dialog("no_memory_found")
                return True

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
        return True

    @intent_handler("RandomMemory.intent")
    def handle_random_memory_intent(self, _message):
        if not self.enabled:
            self.speak_dialog("ntr_disabled_due_to_error")
            return False

        # Show MeeSelf immediately so user knows intent was received
        if self.gui and self.display_mee_image:
            self.gui.show_image(self.image_path, fill='PreserveAspectFit', override_idle=60)

        # Pick a random memory
        memory = random.choice(self.memory_data)
        memory_id = memory["Timestamp"]
        era_name = memory.get("Era", "the past")
        raw_title = memory.get("Title", "")
        spoken_title = self._flip_pronouns(raw_title)

        # PUNCH LIST: Log the title
        self.log.info(f"🎲 Random Memory Selected: {memory.get('Title', 'Untitled')}")

        # 2. INTRODUCE it first (MeePi speaks the title and era)
        # We use wait=True so she finishes this before asking for full/summary
        self.speak_dialog("random_memory", {
            "title": spoken_title,
            "era": era_name
        }, wait=True)

        # memory = results[0]  # Take the first match
        memory_content = self.recall_full_memory(memory_id) or "" # Use timestamp or similar for recall

        if memory_content:
            dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
            self.display_cover_image(memory)  # <== FIXED: pass full dict
            # Using buffered recital for the 'Stop' shield
            self.speak_buffered(dialog_file, memory_content)
            
            # Hand off to Visual Recall
            self.send_visual_recall_request(memory)
            return True  # Fallback Friendly
        else:
            self.log.info(f"RESULTS but NO CONTENT For Random Memory")
            if self.fallback_on:
                return False  # quietly pass on this one Fallback Friendly
            else:
                self.speak_dialog("no_memory_found")
        return False

    @intent_handler("RandomMemoryFromEra.intent")
    def handle_random_memory_from_era_intent(self, message):
        if not self.enabled:
            self.speak_dialog("ntr_disabled_due_to_error")
            return False

        # Show MeeSelf immediately if GUI is available
        if self.gui and self.display_mee_image:
            self.gui.show_image(self.image_path, fill='PreserveAspectFit', override_idle=60)

        # 1. Get query and flip persona: "when you were" -> "when I was"
        raw_query = message.data.get("era", "").lower().strip()
        flipped_query = raw_query.replace("you were", "i was").replace("your", "my")

        # 2. Map to internal Era category
        requested_era = self._map_user_query_to_era(flipped_query)
        self.log.info(f"NTR: Era request '{raw_query}' mapped to '{requested_era}'")

        # 3. Filter memories
        era_matches = [
            m for m in self.memory_data
            if m.get("Era") and requested_era in m.get("Era").lower()
        ]

        if not era_matches:
            self.log.info(f"No memories found for era: {requested_era}")
            # Use raw_query so she says "I don't have memories from when you were..."
            self.speak_dialog("no_memories_for_era", {"era": raw_query})
            return True

        # 4. Pick random and play
        memory = random.choice(era_matches)
        memory_id = memory["Timestamp"]
        era_name = memory.get("Era", "the past")
        spoken_title = self._flip_pronouns(memory.get("Title", ""))

        self.speak_dialog("random_memory", {
            "title": spoken_title,
            "era": era_name
        }, wait=True)

        memory_content = self.recall_full_memory(memory_id) or ""
        if memory_content:
            dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
            self.display_cover_image(memory)
            self.speak_buffered(dialog_file, memory_content)
            self.send_visual_recall_request(memory)
            return True

        return False


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
            if self.gui:
                self.gui.release()

        if self.is_reciting:
            self.is_reciting = False
            self.bus.emit(Message("mycroft.audio.speech.stop"))
            self.speak_dialog("stopped_talking")  # Feedback
            self.log.info("MeePi was interrupted by user.")
