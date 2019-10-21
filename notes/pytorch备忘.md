1. `torch.save(net.state_dict(), 'net.pkl')` 只保存参数会更快   
恢复时需要先再写一遍网络结构然后用 `net2.load_state_dict(torch.load(net.pkl))`
2.  `device = torch.device('cuda:0' if torch.cuda.is_availabel() else 'cpu')   
     data = data.to(device)   
     model.to(device)`   
     这是把模型和数据都送进gpu加速训练