from pytracking.evaluation import Tracker, get_dataset, trackerlist


def eval_atom_cfkd_lasot_trackingnet():
    # run atom_cfkd on LaSOT, TrackingNet, VOT2018
    trackers = trackerlist('atom', 'atom_cfkd')

    dataset = get_dataset('lasot', 'trackingnet')

    return trackers, dataset

def eval_atom_tskd_lasot_trackingnet():
    # run atom_cfkd on LaSOT, TrackingNet, VOT2018
    trackers = trackerlist('atom', 'atom_tskd')

    dataset = get_dataset('lasot', 'trackingnet')

    return trackers, dataset

def eval_atom_compression_lasot_trackingnet():
    # run atom_cfkd on LaSOT, TrackingNet, VOT2018
    trackers = trackerlist('atom', 'atom_compression')

    dataset = get_dataset('lasot', 'trackingnet')

    return trackers, dataset

def eval_drnet_lasot_trackingnet():
    # run atom_cfkd on LaSOT, TrackingNet, VOT2018
    trackers = trackerlist('drnet', 'default')

    dataset = get_dataset('lasot', 'trackingnet')

    return trackers, dataset

def eval_drnet_cfkd_lasot_trackingnet():
    # run atom_cfkd on LaSOT, TrackingNet, VOT2018
    trackers = trackerlist('drnet', 'drnet_cfkd')

    dataset = get_dataset('lasot', 'trackingnet')

    return trackers, dataset