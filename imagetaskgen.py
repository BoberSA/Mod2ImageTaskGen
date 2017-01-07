# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:36:44 2016

@author: Stanislav
"""

# Tasks generator for module 2
# uses numpy, matplotlib, scipy

import random
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy
import scipy.ndimage
import os

# import task text generator
from taskvargen import gen_tasks

# global image size (cols == rows)
rows = 512
cols = rows

# function to get pixel data from figure
def fig2data(fig):
    fig.canvas.draw()
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis = 2)
    return buf

# global colormaps for R, G, B channels
cmaps = [mpl.colors.LinearSegmentedColormap.from_list('',['#000000', '#FF0000']),
         mpl.colors.LinearSegmentedColormap.from_list('',['#000000', '#00FF00']),
         mpl.colors.LinearSegmentedColormap.from_list('',['#000000', '#0000FF']),
         mpl.colors.LinearSegmentedColormap.from_list('',['#000000', '#FFFFFF'])]

def ch_color(ch=0):
    return '#'+''.join(np.roll(list('FF0000'), ch*2))

# shows all channels from buf side-by-side with appropriate colormaps
def show_channels(buf):
    if buf.ndim < 3:
        fig, ax = plt.subplots(figsize = (5,5))
        fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.axis('off')
        ax.margins(0, 0)
        ax.imshow(buf, cmap = cmaps[3])
    else:
        n = buf.shape[2]
        fig, ax = plt.subplots(1, n, figsize = (n*n,n))
        fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        for i in range(n):
            ax[i].xaxis.set_major_locator(plt.NullLocator())
            ax[i].yaxis.set_major_locator(plt.NullLocator())
            ax[i].axis('off')
            ax[i].margins(0, 0)
            ax[i].imshow(buf[:,:,i], cmap = cmaps[i])
    return fig

# apply np.spy to all channels of image
def spy_channels(buf):
    fig, ax = plt.subplots(1, 4, figsize = (16,4))
    labels = ['reds', 'greens', 'blues', 'alpha']
    #fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    for i in range(4):
        ax[i].spy(buf[:,:,i])
        ax[i].set_xlabel(labels[i])

# show RGB|RGBA image or one channel using cmap colormap
def show_image(buf, inch=4, dpi=rows, cmap='jet'):
    row, col = buf.shape[:2]
    fig = plt.figure(figsize = (inch,inch), dpi=dpi/inch)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    fig.set_facecolor('black')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.margins(0, 0)
    ax.axis('off')
    if buf.ndim == 2:
        ax.imshow(buf, cmap=cmap)
    else:
        ax.imshow(buf)
    return fig
    
def gen_noise(low=200, high=220, size=(rows,cols)):
    return np.random.randint(low, high, size, dtype=np.uint8)

# split image into 4 blocks and rotates each block by 90*angles[i] degrees    
def rotate_blocks(buf, angles = (1, 2, 3, 0)):
    s = buf.shape[0] // 2
    for i, a in enumerate(angles):
        r0 = (i // 2) * s
        c0 = (i % 2) * s
        buf[r0:r0+s, c0:c0+s] = np.rot90(buf[r0:r0+s, c0:c0+s], a).copy()
    return buf
    
# split image into 4 blocks and transposes (or not) each block    
def transpose_blocks(buf, trs = (1, 0, 1, 0)):
    s = buf.shape[0] // 2
    for i, t in enumerate(trs):
        if t:
            r0 = (i // 2) * s
            c0 = (i % 2) * s
            buf[r0:r0+s, c0:c0+s] = buf[r0:r0+s, c0:c0+s].T
    return buf
    
# split image into 4 blocks and swaps selected pairs in 'swaps'
def swap_blocks(buf, swaps=((1, 2),(0, 3))):
    s = buf.shape[0] // 2
    tmp = np.zeros((s,s), dtype=buf.dtype)
    for i, sw in enumerate(swaps):
        r0 = (sw[0] // 2) * s
        c0 = (sw[0] % 2) * s
        r1 = (sw[1] // 2) * s
        c1 = (sw[1] % 2) * s
        tmp = buf[r0:r0+s, c0:c0+s].copy()
        buf[r0:r0+s, c0:c0+s] = buf[r1:r1+s, c1:c1+s]
        buf[r1:r1+s, c1:c1+s] = tmp
    return buf

# creates mask for embedding 'X' sign in image
# returns np.array of (s, s) shape and dtype == np.bool
# h - width of sign lines
def makeXmask(s=32, h=3):
    z = np.zeros((s,), dtype=np.bool) | True
    mask = np.diag(z)
    for i in range(1, h//2+2):
        mask |= np.diag(z[i:], i) + np.diag(z[i:], -i)
    mask |= np.rot90(mask)
    return mask
   
# creates mask for embedding 'O' sign in image
# returns np.array of (s, s) shape and dtype == np.bool
# h - width of sign lines
def makeOmask(s=32, h=3):
    s2 = s/2
    x, y = np.mgrid[0:s:1.0, 0:s:1.0]
    r = (x - s2)**2 + (y - s2)**2
    mask = ((s2-h)**2 <= r) & (r < (s2 + 0.5)**2)
    return mask

# creates mask for embedding 'Square' sign in image
# returns np.array of (s, s) shape and dtype == np.bool
# h - width of sign lines
def makeSQmask(s=32, h=3):
    mask = np.zeros((s,s), dtype=np.bool)
    mask[:h*2, :] = True
    mask[:, :h*2] = True
    mask = np.roll(mask, -h, axis=0)
    mask = np.roll(mask, -h, axis=1)
    return mask

# creates mask for embedding 'Triangle' sign in image
# returns np.array of (s, s) shape and dtype == np.bool
# h - width of sign lines
def makeTRImask(s=32, h=3):
    buf = np.zeros((s,s), dtype=np.bool)
    buf[:h, :] = True
    x, y = np.mgrid[0:s:1.0, 0:s:1.0]
    r = y - (-x*(0.5) + s)
    mask = (-h/2<=r) & (r <= h/2)
    buf |= mask
    buf |= np.fliplr(buf)
    buf = np.flipud(buf)
    return buf

# creates mask for embedding 'Arrow' sign in image
# returns np.array of (s, s) shape and dtype == np.bool
# h - width of sign lines
def makeARWmask(s=32, h=3):    
    z = np.zeros((s,), dtype=np.bool) | True
    mask = np.diag(z)
    for i in range(1, h//2+1):
        mask |= np.diag(z[i:], i) + np.diag(z[i:], -i)
    mask = np.roll(mask, s//2, axis=0)
    mask[s//2:,:]=False
    mask |= np.fliplr(mask)
    mask[:, s//2-h//2:s//2+h//2] = True
    return mask
    
# uses mask (for example, created with functions above)
# places sign to one of four blocks using image and mask rotation
def place_sign(buf, mask, v=255, pos=0, delta=(2,2)):
    np.rot90(buf, pos)[delta[0]:mask.shape[0]+delta[0], delta[1]:mask.shape[1]+delta[1]][np.rot90(mask, pos)] = v

# creates noise layer with four signs and four arrows
# arrows are required for visually recognize block direction
def create_noise_layer(low=200, high=220, size=(rows,cols), signv=(200, 255)):
    buf = gen_noise(low, high, size)

    h = 8
    delta = 36
    place_sign(buf, makeSQmask(256-2*delta, h), signv[0], 0, (delta, delta))    
    place_sign(buf, makeTRImask(256-2*delta, h), signv[0], 1, (delta, delta))    
    place_sign(buf, makeOmask(256-2*delta, h), signv[0], 2, (delta, delta))    
    place_sign(buf, makeXmask(256-2*delta, h), signv[0], 3, (delta, delta))

    arw = makeARWmask(24, 4)
    for a in range(4):
        place_sign(buf, arw, signv[1], a, (4, 4))    
    return buf

# uses matplotlib to render text into figure and exports it to np.array    
def create_text_layer(text, size=(rows,cols), v=50):
    buf = np.zeros((rows, cols, 4), dtype=np.uint8)
    buf[:, :, 3] = 255
    plt.ioff()
    fig = show_image(buf)
    ax = fig.axes[0]
    c = '#' + hex(v)[2:].zfill(2) + '0000'
    s = 12
    dy = 3*s
    x0 = 256
    y0 = rows /2 - (len(text)-1)*dy/2
    for i, line in enumerate(text):
        ax.text(x0, y0 + dy * i, line, color=c, fontsize=s, horizontalalignment='center', verticalalignment='top')#, rotation=angle)
#        print(line)
    plt.ion()
    buf1 = fig2data(fig)
    plt.close(fig)
    return buf1[:,:,0]

# uses scipy.ndimage.rotate to rotate image by 'angle' degrees
# return np.array cropped to original shape
def rotate_layer(buf, angle=30):
    r2 = buf.shape[0] // 2
    c2 = buf.shape[1] // 2
    buf1 = scipy.ndimage.rotate(buf, angle)
    r1 = buf1.shape[0] // 2
    c1 = buf1.shape[1] // 2
    return buf1[r1-r2:r1+r2, c1-c2:c1+c2]    

# stacks four layers by third axis in specified order
def stack_layers(R, G, B, A, order='RGBA'):
    d = {'R':R, 'G':G, 'B':B, 'A':A}
    return np.stack((d[order[0]], d[order[1]], d[order[2]], d[order[3]]), axis=2)

# creates various surfaces
def create_surf_layer(surf='abssin(x2+y2)', xy=(2*np.pi, 2*np.pi), size=(rows, cols), params={'dz':20, 'z':10, 'dxy':(np.pi, np.pi)}):
    buf = np.zeros(size, dtype=np.uint8)
    xs = np.linspace(0, xy[0], size[1])
    ys = np.linspace(0, xy[1], size[0])
    x, y = np.meshgrid(xs, ys)

    dxy = params.get('dxy', (0, 0))
    z = params.get('z', 10)
    
    if surf == 'abssin(x2+y2)':
        dz = params.get('dz', 20)
        buf = dz * np.abs(np.sin((x-dxy[0])**2+(y-dxy[1])**2)) + z

    elif surf == 'abssin(|x|+|y|)':
        dz = params.get('dz', 20)
        buf = dz * np.abs(np.sin(np.abs(x-dxy[0])+np.abs(y-dxy[1]))) + z
    
    elif surf == 'abscos(x2+y2)':
        dz = params.get('dz', 20)
        buf = dz * np.abs(np.cos((x-dxy[0])**2+(y-dxy[1])**2)) + z
    
    elif surf == '|x|+|y|+sin2(x+y)':
        k1 = params.get('k1', 1)
        k2 = params.get('k2', 1)
        buf = k1 * (np.abs(x - dxy[0]) + np.abs(y - dxy[1])) + k2 * np.sin(x + y - sum(dxy))**2 + z

    elif surf == '':
        buf = k1 * (np.abs(x - dxy[0]) + np.abs(y - dxy[1])) + k2 * np.sin(x + y - sum(dxy))**2 + z
        
    return np.uint8(buf)

# creates checker (or chess) board
def create_checkerboard(size=128, blocksize=16, low=10, high=20):
    divs = size//blocksize
    bn = np.array(range(blocksize))
    buf = np.zeros((size, size), dtype=np.uint8) + low
    for i in range(2):
        idx = np.hstack(tuple([bn + blocksize*i for i in range(i, divs, 2)]))
        idx = idx.reshape((idx.size, 1))
        buf[idx.T, idx] = high
    return buf

# creates layer with 'n' point on it with value v
# it is very hard to find each single point visually
# therefore, students should write some code to find points    
def create_npoints_layer(n=3, v=255, size=(rows, cols)):
    buf = np.zeros(size)
    added = []
    while (n):
        pt = [random.randint(0, rows-1), random.randint(0, cols-1)]
        if pt not in added:
            buf[pt[0], pt[1]] = v
            n -= 1
            added += [pt]
    return buf

# assumes that weather data loaded from 'stockholm_td_adj.dat'
# from 'Scientific Python Lectures'
# places this data into image layer
def create_weather_layer(data, size=(rows, cols), drow=25, dcol=25, tr=False):
    wbuf = np.zeros((rows, cols), dtype=np.uint8)

    years = set(data[:,0])    
    for i, year in enumerate(years):
        b = (data[:,0] == year)
        c = random.choice([3, 4, 5])
        d = data[b, c] + 30
        dnum = d.shape[0]
        wbuf[i*2 + drow, dcol:dnum+dcol] = d
    if tr:
        return np.uint8(wbuf).T
    return np.uint8(wbuf)
    
    
#def run():
mpl.rc('font', family='Arial')

surfs = ['abssin(x2+y2)', 'abssin(|x|+|y|)', 'abscos(x2+y2)', '|x|+|y|+sin2(x+y)']
data = np.genfromtxt('stockholm_td_adj.dat')

varstart = 0
varnum = 10
ifname = 'tasks_pattern.txt'

# *******************************
tasks = gen_tasks(ifname, varstart, varnum)
# *******************************

tasks_folder = 'tasks'
texts_folder = 'texts'
debug_folder = 'debug'

# make folders if they don't exist
for folder in [tasks_folder, texts_folder, debug_folder]:
    f = os.path.join(os.getcwd(), folder)
    if not os.path.exists(f):
        os.mkdir(f)

ofname = 'tasks.log'
of = open(ofname, 'a')
print('--- start task loop ---')
for task in tasks:
    parms = task[0]
    l1 = parms['layer']
    chl = parms['lnum']
    text = [line.strip() for line in task[1].split('\n')]
    noise_ch = create_noise_layer(180, 220, signv=(210, 220))
    rots = [1, 1, 2 ,3]
    trs = [0, 0, 1, 1]
    random.shuffle(rots)
    random.shuffle(trs)
    noise_ch1 = noise_ch.copy()
    rotate_blocks(noise_ch1, rots)
    noise_ch2 = noise_ch1.copy()
    transpose_blocks(noise_ch2, rots)
    swaps = [0, 1, 2, 3]
    random.shuffle(swaps)
    swap_blocks(noise_ch2, [(swaps[0], swaps[1])])

    text_ch = create_text_layer(text, v=5)
    text_ch1 = rotate_layer(text_ch, random.choice(range(-30, 31)))
    text_ch2 = text_ch1 + noise_ch2
    
    sprm = {'k1':random.choice(range(1, 4)),
            'k2':random.choice(range(1, 4)), 
            'z':random.choice(range(50, 150)),
            'dz':random.choice(range(20, 50)),
            'dxy':(random.random() * chl, random.random() * chl)}
    surfs_ = surfs[:]
    sname = random.choice(surfs)
    surfs_.remove(sname)
    surf_ch = []
    surf_ch.append(create_surf_layer(sname, (chl, chl), params=sprm))    
    
    if (parms['task_type'] == 0):
        l2 = parms['layer2']
        sprm = {'k1':random.choice(range(1, 4)),
                'k2':random.choice(range(1, 4)), 
                'z':random.choice(range(50, 150)),
                'dz':random.choice(range(20, 50)),
                'dxy':(chl/2, chl/2)}
        sname = random.choice(surfs)
        surf_ch.append(create_surf_layer(sname, (chl, chl), params=sprm))

    elif (parms['task_type'] == 1):
        bsize = parms['size']
        l2 = parms['layer2']
        sh = 2
        low = random.choice(range(0, 32))
        high = random.choice(range(64, 96))
        chb_buf = create_checkerboard(rows, bsize, low, high)
        sign_lst = []
        i = 3
        while i != 0:
            sign_x = random.choice(range(512//bsize))
            sign_y = random.choice(range(512//bsize))
            if (sign_x, sign_y) in sign_lst:
                continue
            sign_lst += [(sign_x, sign_y)]
            i -= 1
            fun = random.choice([makeXmask, makeOmask, makeTRImask, makeARWmask])
            mask = fun(bsize-sh*2, 3)
            place_sign(chb_buf, mask, 128, 0, (sign_x*bsize+sh, sign_y*bsize+sh))
        surf_ch.append(chb_buf)
        
    elif (parms['task_type'] == 2) or (parms['task_type'] == 3):
        l2 = parms['layer2']
        surf_ch.append(create_npoints_layer(v=random.randint(1, 255)))
        
    elif (parms['task_type'] == 4):
        l2 = parms['layer2']
        tr = parms['tr']
        drow = random.randint(0, 50)
        dcol = random.randint(0, 100)
        weather_ch = create_weather_layer(data, drow=drow, dcol=dcol, tr=tr)
        surf_ch.append(weather_ch)
       
    d = {}
    d[l1] = surf_ch[0]
    d[l2] = surf_ch[1]

    chs2 = list('RGBA')
    chs2.remove(l1)
    chs2.remove(l2)
    random.shuffle(chs2)
    d[chs2[0]] = noise_ch
    d[chs2[1]] = text_ch2 
    buf = stack_layers(**d)
    # gen 'almost random' file name
    n1 = hex(parms['task_id'])[2:].zfill(2)
    n2 = hex(random.randint(0, 100))[2:].zfill(2)
    n3 = hex(random.randint(0, 100))[2:].zfill(2)
    fnum = n2 + n1 + n3
    
    wfname = os.path.join(tasks_folder, 'task_' + fnum + '.png')
    wtxtname = os.path.join(texts_folder, 'text_' + fnum + '.png')
    
    # save task image
    img.imsave(wfname, buf)
    # save image with task text
    img.imsave(wtxtname, text_ch1, cmap=cmaps[3])
    # save task text ot log
    of.write(wtxtname + '\n' + task[1] + '\n')
    
    # save image with expanded channels for debug
    fig = show_channels(buf)
    dbgfname = os.path.join(debug_folder, 'dbg_' + fnum + '.png')
    fig.savefig(dbgfname)
    plt.close(fig)
    
    print('id:', parms['task_id'], 'type:', parms['task_type'])
    
print('--- end task loop ---')
of.close()
#    return 0

# working in global namespace because of simple debugging in Spyder
# (like in Matlab)
## --- main ---
#if __name__ == '__main__':
#    run()
    