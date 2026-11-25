# Raw traces for PL1

- nn_random_batch.csv: network (SPOCK DeepRegressor) on 1,750 random-set systems: index, log10 median, lower, upper.
- feat_random_a.csv, feat_random_b.csv (this workspace) and all_feats.csv (Colab: set, index, ...): short-integration summaries.
- idx_resonant.txt: resonant training sample; random_split.npy: DEV (True) / FINAL split of the random set.
- fresh_systems.bin: the 800 fresh systems (REBOUND Simulationarchive); fresh_ic.npz: their elements and source indices.
- fresh_nbody_local.csv, fresh_nbody_colab.csv: fresh N-body truth (index, t/P1, survived-to-1e6 flag).
- featfresh_all.csv, nnfresh_all.csv: summaries and network predictions for the fresh systems.
