import os

path = r'C:\Users\dana\Documents\Ido\follow_up_project\datasets\efi\labels\fused'
destpath = r'C:\Users\dana\Documents\Ido\follow_up_project\datasets\efi\labels\fused_rect'
if not os.path.exists(destpath):
    os.mkdir(destpath)
for filename in os.listdir(path):
    if filename.endswith(".txt"):
        with open(os.path.join(path, filename), "r") as f:
            lines=f.readlines()
        with open(os.path.join(destpath, (filename)), "w") as g:
            for i, line in enumerate(lines):
                # if i == 0:
                #     g.write(line)
                # else:
                log = [float(x) for x in line.split()]
                x=log[0]
                y=log[1]
                r=log[2]
                # sc = log[3]
                x1 = int(x-r)
                x2 = int(x+r)
                y1=int(y-r)
                y2=int(y+r)
                g.write(('%g ' * 4 + '\n') % (x1, y1, x2, y2))
                # g.write(('%g ' * 5 + '\n') % (x1, y1, x2, y2, sc))