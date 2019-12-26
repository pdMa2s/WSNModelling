from dataGeneration.networks import Fontinha, Richmond

if __name__ == '__main__':
    # rich = Richmond()
    # time_incs = rich.generate_sim_data()
    # print(time_incs)
    fontinha = Fontinha()
    time_incs = fontinha.generate_sim_data(n_batches=5000)
    time_incs = time_incs.round({key: 2 for key in time_incs.columns})
    print(time_incs)
    print("Saving the data...")
    time_incs.to_csv("../fontinha_data.csv", index=False)