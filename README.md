# Near Total Recall Skill

MeePi Memory Recall Skill.  Much More to come.  May or may not be useful to others

## Overview

The Near Total Recall Skill enables you to retrieve and recall specific topics (memories) from a dataset using natural language queries. By leveraging advanced sentence embeddings and cosine similarity, this skill identifies and presents the most relevant memories based on your input.

## Features

**Memory Retrieval**: Efficiently find/recite memories related to queries.
**Similarity Thresholds**: Configurable thresholds to control the sensitivity of memory matches.
**Response Templates**: Customizable templates for presenting recalled memories.

## Installation/Configuration

This contraption is complicated and in development ... it involves separate data collection, a Cleaning/ML Pipeline on Colab & some other stuff PLUS I'm just learning all this stuff!
Probably best to stay away from this code at this time

OVOS Settings
{
    "__mycroft_skill_firstrun": false,
    "cleaned_data_path": "<Path to Cleaned/Summarized Memories>",
    "embeddings_path": "<Path to Memories Vector Embeddings file (.npy)>",
    "original_data_path": "<Path to TTS-Friendly Memories file (.csv)",
    "top_n": 5,
    "similarity_threshold": 0.5,
    "model_name": "all-MiniLM-L6-v2"
}


## Examples

TBD ... 
- "Do you remember {topic}"
- "Do you recall {topic}"
- "Tell me about {topic}" ...."

## Credits

Tom P (@tped)

## Category

TODO:  MUCH TO DO!

## Tags

ovos skill
