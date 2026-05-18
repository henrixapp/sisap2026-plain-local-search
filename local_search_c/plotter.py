import sys
import matplotlib.pyplot as plt

def main(files,out1):
    # 1. Turn on Interactive Mode for live updating
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Quality')
    ax.set_title('Time vs. Quality (Live Updates)')
    ax.grid(True)
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    markers = ['o', 'x', 's', '^', 'v', '<', '>']
    color_idx = 0
    
    # Helper to create a new line on the graph
    def add_line(label):
        nonlocal color_idx
        c = colors[color_idx % len(colors)]
        m = markers[color_idx % len(markers)]
        color_idx += 1
        # Create an empty line object that we can update later
        line, = ax.plot([], [], label=label, marker=m, linestyle='-', color=c)
        return line

    # 2. Process any static files passed via command line first
    for file in files:
        print(f"Reading file: {file}")
        try:
            with open(file, 'r') as f:
                q, t, gens = parse_static_stream(f)
                if q and t:
                    line = add_line(file)
                    line.set_xdata(t)
                    line.set_ydata(q)
                # for g in gens:
                #     plt.vlines([float(g)],[0],[1],label=file)
        except FileNotFoundError:
            print(f"Error: File not found: {file}")
            
    # Draw the initial plot for the files
    ax.relim()
    ax.autoscale_view()
    if len(files) > 0:
        ax.legend()
    plt.draw()
    plt.pause(0.1) # Force GUI update
    
    # 3. Process Standard Input dynamically (Live Update)
    print("Reading from standard input. Waiting for START_MESSAGE...")
    stdin_q = []
    stdin_t = []
    start_parsing = False
    stdin_line = None
    
    # sys.stdin acts as a generator, yielding lines as they are received
    with open(out1,"w")as f:
        while True:
        # readline() blocks cleanly until a '\n' arrives, then jumps into action
            line_str = sys.stdin.readline()
            line_str = line_str.strip()
            print(line_str)
            
            if line_str == 'START_MESSAGE':
                f.write("START_MESSAGE\n")
                
                start_parsing = True
                if stdin_line is None:
                    stdin_line = add_line(out1)
                    ax.legend()
                continue
            if line_str =="STOP_MESSAGE":
                break   
            if start_parsing and line_str:
                parts = line_str.split()
                if "-1" in line_str:
                    plt.vlines([float(parts[1])],[0],[1],label=out1)
                if len(parts) >= 2:
                    try:
                        q_val = float(parts[0])
                        t_val = float(parts[1])
                        f.write(f"{q_val} {t_val} {parts[2] if len(parts)>2 else ""}\n")
                        f.flush()
                        
                        stdin_q.append(q_val)
                        stdin_t.append(t_val)
                        
                        # Update the live line data
                        stdin_line.set_xdata(stdin_t)
                        stdin_line.set_ydata(stdin_q)
                        
                        # Rescale axes to fit new data
                        ax.relim()
                        ax.autoscale_view()
                        
                        # plt.pause redraws the canvas and processes GUI events
                        plt.pause(0.01) 
                        
                    except ValueError:
                        pass
    
        
    # 4. Standard Input closed (EOF reached)
    print("\nInput stream finished.")
    plt.savefig('live_parsed_plot.png')
    print("Plot saved successfully as live_parsed_plot.png")
    
    # Turn off interactive mode and block so the window doesn't instantly close
    plt.ioff()
    plt.show()

def parse_static_stream(stream):
    """Helper to parse static files all at once."""
    quality = []
    time = []
    gens = []
    start_parsing = False
    for line in stream:
        line = line.strip()
        if line == 'START_MESSAGE':
            start_parsing = True
            continue
        if start_parsing and line:
            parts = line.split()
            if "-1" in line:
                gens += [float(parts[1])]
            if len(parts) >= 2:
                try:
                    quality.append(float(parts[0]))
                    time.append(float(parts[1]))
                except ValueError:
                    pass
    return quality, time, gens

if __name__ == "__main__":
    input_files = sys.argv[1:len(sys.argv)-1]
    output = sys.argv[-1]
    main(input_files,output)