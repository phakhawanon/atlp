from .motion import (
    joint_lim_dict,        
)


import matplotlib.pyplot as plt


# file_path = 'data.json'
# report_file_path = 'report.txt'

def plot_joint_states(time_array, joint_arrays, subfields):
    """
        Plot joint states
    """    
    fig, axes = plt.subplots(len(subfields), 1, sharex=True, figsize=(5, len(subfields)*2.2))
    duration = time_array[-1] - time_array[0]
   
    if len(subfields) == 1:
        enumerator = list()
        enumerator.append(axes)
    else:
        enumerator = axes.flat
    for i, ax in enumerate(enumerator):
        ax.plot(time_array, joint_arrays[i])
        ax.set_xlim(0.0, duration)
        lim_lower, lim_upper = joint_lim_dict[subfields[i]]
        ax.set_ylim(lim_lower*1.2, lim_upper*1.2)
        ax.axhline(y=lim_lower, linestyle=":", color="red")
        ax.axhline(y=lim_upper, linestyle=":", color="red")
        ax.plot(time_array,joint_arrays[i], color="blue")
        ax.set_title(subfields[i])
        ax.grid()
    
    plt.tight_layout()
    fig.savefig("test_joint_plot.jpg")

# def main():
        
#     global data

#     with open(file_path,'r') as f:
#         data = json.load(f)
    
#     # Get min and max timestamps of the dataset
#     min_timestamp, max_timestamp = analyze_timestamp()
#     print((min_timestamp, max_timestamp))

#     # Calculate their difference
#     diff = max_timestamp - min_timestamp
#     print(diff)

#     # Get duration in s
#     duration = get_video_duration()
#     print(duration)

#     plot_field("state_right_arm_joint_position", min_timestamp, diff, duration)


if __name__ == "__main__":
    # main()
    print("Visualizer module is loaded successfully.")
