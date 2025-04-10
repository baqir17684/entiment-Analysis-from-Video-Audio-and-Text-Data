from mmsdk import mmdatasdk
import pickle
local_recipe = {
    "glove_vectors": "cmumosei/CMU_MOSEI_TimestampedWordVectors.csd",
    "COVAREP": "cmumosei/CMU_MOSEI_COVAREP.csd",
    "OpenFace_2": "cmumosei/CMU_MOSEI_VisualOpenFace2.csd",
    "FACET 4.2": "cmumosei/CMU_MOSEI_VisualFacet42.csd",
}

dataset = mmdatasdk.mmdataset(local_recipe)

labels_recipe = {
    "All Labels": "cmumosei/CMU_MOSEI_Labels.csd",
}
dataset.add_computational_sequences(labels_recipe, 'cmumosei/')

print("Starting alignment...")

dataset.align('All Labels')

with open("aligned_mosei.pkl", "wb") as f:
    pickle.dump(dataset, f)

