from ovos_utils import classproperty
from ovos_utils.process_utils import RuntimeRequirements
from ovos_workshop.decorators import intent_handler
from ovos_workshop.skills import OVOSSkill

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


# NTR data and tuning parameters in <NTR_Skill>/settings.json
DEFAULT_SETTINGS = {
    "cleaned_data_path": "/home/ovos/NTR-Data/cleaned_Memories.csv",
    "embeddings_path": "/home/ovos/NTR-Data/MeePi_embeddings.npy",
    "original_data_path": "/home/ovos/NTR-Data/MeePiMemories.csv",
    "image_path": "/home/ovos/MeePi-Media/cover.jpg",
    "display_image":  True,

    # Tuning parameters (from CONFIG in your Python script)
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
        self.is_reciting = False  # Track if MeePi is currently babbling

        # Move here suggested by AI
        # self.settings.merge(DEFAULT_SETTINGS, new_only=True)

        # Load settings from self.settings
        self.cleaned_data_path = self.settings.get("cleaned_data_path")
        self.embeddings_path = self.settings.get("embeddings_path")
        self.original_data_path = self.settings.get("original_data_path")
        self.image_path = self.settings.get("image_path")

        self.display_image = self.settings.get("display_image")
        self.top_n = self.settings.get("top_n")
        self.similarity_threshold = self.settings.get("similarity_threshold")
        self.model_name = self.settings.get("model_name")

        # Initialize with paths to the cleaned data and embeddings.
        try:
            self.cleaned_data = pd.read_csv(self.cleaned_data_path)
        except Exception as e:
            self.log.error(f"Failed to load cleaned data: {e}")
            self.cleaned_data = None  # Prevents crashes later

        try:
            self.embeddings = np.load(self.embeddings_path)
        except Exception as e:
            self.log.error(f"Failed to load embeddings: {e}")
            self.embeddings = None

        try:
            self.original_data = pd.read_csv(self.original_data_path)
        except Exception as e:
            self.log.error(f"Failed to load original data: {e}")
            self.original_data = None

        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            self.log.error(f"Failed to load Sentence Transformer model: {e}")
            self.model = None

        # Notify the user if something went wrong
        if None in [self.cleaned_data, self.embeddings, self.original_data, self.model]:
            self.speak_dialog("error_initialization")

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
        if self.cleaned_data is None or self.embeddings is None:
            self.log.error("Cleaned data or embeddings not loaded.")
            return []

        # Use the model to encode the query
        query_embedding = self.model.encode([query])

        # Compute similarity between query and all memory embeddings
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()

        # Find the top N most similar memories
        top_n_indices = np.argsort(similarities)[::-1][:self.top_n]
        results = [(similarities[i], self.cleaned_data.iloc[i], self.cleaned_data.iloc[i]['Timestamp']) for i in
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
        if self.original_data is None or self.cleaned_data is None:
            self.log.error("Original data not loaded.")
            return None

        # Assuming memory_id corresponds to the 'Timestamp' or another unique field
        memory_row = self.original_data[self.original_data['Timestamp'] == memory_id]
        cleaned_row = self.cleaned_data[self.cleaned_data['Timestamp'] == memory_id]

        if memory_row.empty:
            return None  # Memory not found

        # Extract details
        description = memory_row.iloc[0]['Memory_Description']
        is_long = cleaned_row.iloc[0].get("is_long_story", False) if not cleaned_row.empty else False
        has_summary = "Memory_Summary" in cleaned_row and not pd.isna(cleaned_row.iloc[0].get("Memory_Summary", None))

        # Looks Like we will speak - display MeePi image
        if self.display_image:
            self.gui.show_image(self.image_path, fill='PreserveAspectFit')

        # Warn the user and offer summary if available
        if is_long:
            response = self.get_response("long_story_warning")  # Ask user for choice
            if response and "summary" in response.lower() and has_summary:
                return cleaned_row.iloc[0]["Memory_Summary"]  # Return summary

        return description  # Default to full memory

    @intent_handler("DoYouRecall.intent")
    def handle_do_you_recall_intent(self, message):
        query = message.data.get("query", "")
        self.log.info(f"Received query for recall: {query}")

        # Check for exact title match (case_insensitive)
        exact_match = self.cleaned_data[self.cleaned_data['Title'].str.lower() == query.lower()]
        if not exact_match.empty:
            memory_content = self.recall_full_memory(exact_match.iloc[0]['Timestamp'])
            if memory_content:
                dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
                self.is_reciting = True
                self.speak_dialog(dialog_file, {"memory": memory_content}, wait=True)
                self.is_reciting = False
                return True  # Fallback Friendly 3
            else:
                # self.speak_dialog("no_memory_found")
                return False  # Fallback Friendly 3
            # return  # Early return on exact match Removed for Fallback Friendliness

        # Fallback to the closest match logic
        results = self.find_closest_memory(query)

        # Handle results
        if results:
            memory = results[0]  # Take the first match
            memory_content = self.recall_full_memory(memory[2])  # Use timestamp or similar for recall

            if memory_content:
                dialog_file = "recite_memory" if len(memory_content.split()) > 20 else "recite_summary"
                self.is_reciting = True
                self.speak_dialog(dialog_file, {"memory": memory_content}, wait=True)
                self.is_reciting = False
                return True  # Fallback Friendly 3
            else:
                # self.speak_dialog("no_memory_found")
                return False  # Fallback Friendly 3
        else:
            # self.speak_dialog("no_memory_found")
            return False  # Fallback Friendly 3

    def stop(self):
        """ Action to take when "stop" is requested by the user.
        """
        if self.is_reciting:
            self.speak("")  # Stop MeePi from talking
            self.is_reciting = False
            self.speak_dialog("stopped_talking.dialog")  # Feedback
            self.log.info("MeePi was interrupted by user.")
            return True  # Indicate that MeePi stopped
        return  # Nothing was interrupted
