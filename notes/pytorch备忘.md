1. `torch.save(net.state_dict(), 'net.pkl')` 只保存参数会更快   
恢复时需要先再写一遍网络结构然后用 `net2.load_state_dict(torch.load(net.pkl))`