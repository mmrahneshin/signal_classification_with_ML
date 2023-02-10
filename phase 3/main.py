from data_preparation.data_preparation import data_preparation

def main():
    data, label = data_preparation()

    print(data.shape, label.shape)

main()