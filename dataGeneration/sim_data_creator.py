from dataGeneration.networks import Fontinha, Richmond

if __name__ == '__main__':
    #for network in [Fontinha(), Richmond()]:
    for network in [Richmond(sim_step=3600, hydraulic_step=30)]:
        time_incs = network.generate_sim_data(n_batches=2000)
        time_incs = time_incs.round({key: 2 for key in time_incs.columns})
        print(time_incs)
        print("Saving the data...")
        time_incs.to_csv(f"{network.name.lower()}_data.csv", index=False)