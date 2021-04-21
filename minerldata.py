import minerl
import pickle
import gzip
import os
import numpy as np
import torch as T

def get_minerl_dataset(size = 20000, envname = "Treechop", mode = "full"):
    datadir = "data/minerl/"   
    filepath = datadir + f"{envname}-{mode}-{size}.pickle"

    # CHECK IF ALREADY EXISTS
    if os.path.exists(filepath):
        print("loading dataset...")
        with gzip.open(datadir + f"{envname}-{mode}-{size}.pickle", 'rb') as fp:
            X = pickle.load(fp)
        print("finished loading dataset")
        return T.from_numpy(X).permute(0,3,1,2)

    os.makedirs(datadir, exist_ok=True)
    if not os.path.exists(f"{os.getenv('MINERL_DATA_ROOT', 'data/')}/MineRL{envname}VectorObf-v0"):
        minerl.data.download(os.getenv('MINERL_DATA_ROOT', 'data/'), experiment=f'MineRL{envname}VectorObf-v0')
    data = minerl.data.make(f'MineRL{envname}VectorObf-v0', data_dir=os.getenv('MINERL_DATA_ROOT', 'data/'),
                            num_workers=1, worker_batch_size=1)
    names = data.get_trajectory_names()

    #np.random.default_rng().shuffle(names)
    X = np.zeros((size, 64, 64, 3), dtype=np.uint8)
    print(f"collecting {envname} data set with", size, "frames")

    # DEV
    full_ep_lens = 0

    runidx = 0
    for name_idx, name in enumerate(names):
        #print(name)
        print("percentage of episodes used so far:", round(name_idx/len(names)*100),
                "dataset size:", runidx,
                "full ep lens:", full_ep_lens)
        # EXTRACT EPISODE
        state, action, reward, _, done = zip(*data.load_data(name))
        pov = np.stack([s['pov'] for s in state])
        add = min(size-runidx, len(pov))
        
        #reward = np.array(reward[:add])
        # get full ep len of all
        full_ep_lens += len(pov)

        X[runidx:runidx+add] = pov[:add]

        runidx += add
        if runidx >= size:
            break

    # SAVE AS ZIPPED FILE
    with gzip.GzipFile(filepath, 'wb') as fp:
        pickle.dump((X[:runidx]), fp)

    # DEV
    print("full ep length:", full_ep_lens, "beginning percentage", size/full_ep_lens)

    return T.from_numpy(X).permute(0,3,1,2)