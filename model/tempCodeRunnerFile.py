config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    model = Mob(chronos_name=config['mobtep']['chronos_name'], 
                internvl_name=config['mobtep']['internvl_name'],
                projector_init=config['mobtep']['projector_init'],
                sum_ts_emb_to=config['mobtep']['sum_ts_emb_to']).to(device)
    
    checkpoint_path = f"/home/ubuntu/thesis/model/checkpoints/{config['mobtep']['mob_checkpoint']}.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    ts_folder_path = "/home/ubuntu/thesis/data/samples/new samples no overlap/test/time series"
    metadata_folder_pth = "/home/ubuntu/thesis/data/samples/new samples no overlap/test/metadata"
    image_folder_path = "/home/ubuntu/thesis/data/samples/new samples no overlap/test/plots"
    save_folder_path= f"/home/ubuntu/thesis/data/samples/new samples no overlap/generated captions/internvl_{config['mobtep']['mob_checkpoint']}"
    
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    
    
    
    generate_captions(model, ts_folder_path, metadata_folder_pth, image_folder_path, save_folder_path, batch_size=15, use_chronos=config['mobtep']['use_chronos'])