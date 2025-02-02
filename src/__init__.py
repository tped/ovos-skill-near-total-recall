from ovos_utils import classproperty
from ovos_utils.process_utils import RuntimeRequirements
from ovos_workshop.intents import IntentBuilder
from ovos_workshop.decorators import intent_handler
# from ovos_workshop.intents import IntentHandler # Uncomment to use Adapt intents
from ovos_workshop.skills import OVOSSkill
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


# NTR data and tuning parameters in <NTR_Skill>/settings.json
DEFAULT_SETTINGS = {
    "cleaned_data_path": "/path/to/cleaned_memories.csv",
    "embeddings_path": "/path/to/memory_embeddings.npy",
    "original_data_path": "/path/to/MeePiMemories.csv",

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

        # Load settings from self.settings
        self.cleaned_data_path = self.settings.get("cleaned_data_path")
        self.embeddings_path = self.settings.get("embeddings_path")
        self.original_data_path = self.settings.get("original_data_path")

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

        return results

    def recall_full_memory(self, memory_id):
        """
        This method retrieves the full memory details (e.g., from MeePiMemories.csv) using the memory ID.
        """
        if self.original_data is None:
            self.log.error("Original data not loaded.")
            return None

        # Assuming memory_id corresponds to the 'Timestamp' or another unique field
        memory_row = self.original_data[self.original_data['Timestamp'] == memory_id]

        if not memory_row.empty:
            # Construct a full memory description
            full_memory = memory_row.iloc[0]['Memory_Description']
            return full_memory
        else:
            return None

    @intent_handler("DoYouRecall.intent")
    def handle_do_you_recall_intent(self, message):
        query = message.data.get("query", "")
        self.log.info(f"Received query for recall: {query}")
        # Assuming you have a find_closest_memory method
        results = self.find_closest_memory(query)

        # Handle results
        if results:
            memory = results[0]  # Take the first match
            full_memory = self.recall_full_memory(memory[2])  # Use timestamp or similar for recall
            self.speak_dialog("recite_memory", {"memory": full_memory})
        else:
            self.speak_dialog("no_memory_found")

    @intent_handler(IntentBuilder("RoboticsLawsIntent").require("LawKeyword").build())
    def handle_robotic_laws_intent(self, message):
        """This is an Adapt intent handler, but using a RegEx intent."""
        """This is an Adapt intent handler, it is triggered by a keyword.
                Skills can log useful information. These will appear in the CLI and
                the skills.log file."""
        self.log.info("There are five types of log messages: " "info, debug, warning, error, and exception.")
        # Optionally, get the RegEx group from the intent message
        # law = str(message.data.get("LawOfRobotics", "all"))
        self.speak_dialog("robotics")

    def stop(self):
        """Optional action to take when "stop" is requested by the user.
        This method should return True if it stopped something or
        False (or None) otherwise.
        If not relevant to your skill, feel free to remove.
        """
        return
