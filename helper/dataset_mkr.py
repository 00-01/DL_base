def v0_9_df2tensor(dataframe):
    data = []
    label = []
    for index, row in dataframe.iterrows():
        try:
            img = Image.open(row[col[0]])
            data.append(list(img.getdata()))
            label.append(row[col[1]])
        except Exception as E:
            print(E)
        if index%10000 == 0:  print(index)

    data = np.array(data)
    data = data.reshape(data.shape[0], H, W, 1)

    label = np.array(label)
    label = label.reshape(label.shape[0], 1)

    return data, label

