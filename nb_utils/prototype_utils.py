def scatter_plot_base_separate(samples, samples_before, 
                               targets, targets_before, classes=[0,1,2], 
                               base_classes_k=0, show=True,
                              show_classes=False,
                              dataset='train'):
    scaler1 = StandardScaler()
    samples_normalized = scaler1.fit_transform(samples)
    scaler2 = StandardScaler()
    samples_before_normalized = scaler2.fit_transform(samples_before)
    
    pca1 = decomposition.PCA(n_components=2)
    pca1.fit(samples_normalized)
    points = pca1.transform(samples_normalized)
    
    print('explained variance ', pca1.explained_variance_ratio_)
    
    pca2 = decomposition.PCA(n_components=2)
    pca2.fit(samples_before_normalized)

#     print('explained variance ', pca2.explained_variance_ratio_)

    
    marker_list = ['<', 's', 'P', 'd']
    fig, ax = plt.subplots()
    colors_list = list(mcolors.TABLEAU_COLORS.keys())
#     print('classes ', classes)
    top_k_list = []
    for t, i in enumerate(classes):
#         i = classes[t]
        color = colors_list[t]
        class_points = points[targets==i,:]
#         print('LENNN ', len(class_points))
        # get position of protos
        class_proto = np.expand_dims(samples_normalized[targets==i, :].mean(0), 0)

        class_mean_point = pca1.transform(class_proto).squeeze()
#         print(get_class(i, dataset), color[len('tab:'):])


        x = class_points[:, 0]
        y = class_points[:, 1]
        scale = 50.0 
        label = get_class(i, dataset)
        ax.scatter(x[:1000], y[:1000], c=color, s=scale, label=label,
                   alpha=0.3, edgecolors='none')
        ax.scatter(class_mean_point[0], class_mean_point[1], s=100.0, label=label,
                   alpha=1, edgecolors='none')
        
        # now plot base_classes
        # get top k closest to samples_before
        class_proto = np.expand_dims(samples_before[targets_before==i, :].mean(0), 0)
#         print('class_proto ', class_proto.shape)
        top_k, wts = get_k_base(class_proto, all_proto, k=base_classes_k, return_weights=True)
        top_k = top_k.squeeze()
        wts = np.array(wts.squeeze())

        top_k_list.append(top_k)
#         print('top_k ', top_k)
#         print('wts ', wts)
        base_proto = all_proto[top_k, :].squeeze()
#              

        for t, proto in enumerate(base_proto):
            print([get_class(top_k[t], 'train'), marker_list[t]])
            proto = np.expand_dims(proto, 0)
            proto_norm = (proto-scaler2.mean_)/np.sqrt(scaler2.var_+1e-10) # this has to be scaler2 because normalization won't be correct otherwise?
            class_mean_point = pca1.transform(proto_norm).squeeze()
#             print('class_mean ', class_mean_point)
            label = get_class(top_k[t], 'train') + '{:0.4f}'.format(wts[t])
            ax.scatter(class_mean_point[0], class_mean_point[1], s=100.0, label=label,
            alpha=1, edgecolors='none', marker=marker_list[t], color=color)

#     ax.legend()

    ax.legend(bbox_to_anchor=(1.05,1.0))

    ax.grid(True)
    if show:
        plt.show()
    else:
        return ax

    if show_classes:
        for c in classes:
#             print('dataset ', dataset)
#             print(get_class(c, dataset))
            plot_class(c, dataset)
        top_k_list = np.concatenate(top_k_list)
        for c in top_k_list:
#             print(get_class(c, 'train'))
            plot_class(c, trainset)
        


def get_samples(trainer, loader, n=200):
    model = trainer.model.eval()
    samples_before = []
    samples = []
    samples_feat = []
    targets = []
    targets_feat = []
    for i in range(n):
        data, target = next(iter(loader))
        with torch.no_grad():
            print('i', i)
            print(data.shape)
            data = data.cuda()
            features = model.encoder(data)
            features = features.unsqueeze(0)
            after_attn = model.slf_attn(features, features, features).squeeze()
            samples_before.append(features.detach().cpu().squeeze().numpy())
            samples.append(after_attn.detach().cpu().numpy())
            targets.append(target.numpy())

            # feat;
            _ = model(data)
            print('after_attn shape', model.after_attn.shape)
            samples_feat.append(model.after_attn.squeeze().detach().cpu().squeeze().numpy())
            targets_feat.append(target[:args.eval_way*args.eval_shot].numpy())
    samples_before = np.concatenate(samples_before)
    samples = np.concatenate(samples)
    targets = np.concatenate(targets)

    samples_feat = np.concatenate(samples_feat)
    targets_feat = np.concatenate(targets_feat)
    return samples_before, samples_feat, targets, targets_feat