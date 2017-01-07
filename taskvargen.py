# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 00:43:48 2017

@author: Stanislav Bober
"""

import random
import numpy as np

# algorithms for simple sequence generation
# generates i-th sequence member

# def cnt_repeat - main algorithm
# generates sequences like:
# 1, 1, ... 1, 2, 2, ... 2, 3, 3, ..., 3 ...
# |<-  r  ->|  |<-  r  ->|  |<-  r  ->|
#
# trivial usage:
# cnt_repeat(i) generates
# i:      0 1 2 3 4 5 6 7 8
# return: 1 2 3 1 2 3 1 2 3
#
def cnt_repeat(i, lst=[1, 2, 3], r=1):
    j = (i // r) % len(lst)
    return lst[j]

# cnt_updown - second algorithm
# generates sequences like:
# 1, 2, 3, 4, 4, 3, 2, 1 ...
# 1, 2, 3, 4, 3, 2, 1, 2 ...
#
def cnt_updown(i, lst=[1, 2, 3, 4], tops=True):
    if tops:
        return cnt_repeat(i, lst + lst[::-1]) # 1, 2, 3, 4, 4, 3, 2, 1 ...
    return cnt_repeat(i, lst + lst[-2:0:-1])  # 1, 2, 3, 4, 3, 2, 1, 2 ...

# cnt_random - returns random choice from lst
#
def cnt_random(lst=[1, 2, 3]):
    return random.choice(lst)

# dict with algorithm names
#
algs = {'ord' :lambda i, lst: cnt_repeat(i, lst, 1),
        'rep2':lambda i, lst: cnt_repeat(i, lst, 2),
        'rep3':lambda i, lst: cnt_repeat(i, lst, 3),
        'ud'  :lambda i, lst: cnt_updown(i, lst),
        'ud_' :lambda i, lst: cnt_updown(i, lst, False),
        'rnd' :lambda i, lst: cnt_random(lst)}

# def get_alg
# dct - dict with task parameters
# it can be one of two types:
# 1. {'param0':values0, 'param1':values1, ...}
# 2. {'alg':'alg_name', 'vals':{'param0':values0, 'param1':values1, ...}}
#
def get_alg(dct):
    if type(dct) == dict:
        if 'alg' in dct:
            if dct['alg'] in algs:
                fn = algs[dct['alg']]
                return lambda i : fn(i, dct['vals'])
            else:
                return lambda i: algs['ord'](i, dct['vals'])

    return lambda i: algs['ord'](i, dct)

# def substitute
#
# this function generates dict with actual parameters for task with global
# number i and then insert this parameters into template of task text
# applying .format function available for any string
# returns [dict_of_actual_params, task_text]
#
# i - global task number
# tasks - templates of task texts (texts with embedded parameters)
# values - list of dicts with individual parameters for all tasks
# common - dict with common parameters for all tasks
#
def substitute(i, tasks, values, common):   
    task_num = i % len(tasks)
    parms = {'task_id':i, 'task_type':task_num}
    # append all common params
    for p in common:
        fn = get_alg(common[p])
        parms[p] = fn(i)    
    # append local task params
    task_vals = values[task_num]
    for p in task_vals:
        fn = get_alg(task_vals[p])
        parms[p] = fn(i)
    task = tasks[task_num]
    task = task.format(**parms)
    return [parms, task]

# def gen_tasks
# loads 'fname' file with templates of task texts and parameters dicts
# generates 'n' task texts starting from 'start' global number 
# returns list of [dict_of_actual_params, task_text] for each task
#
def gen_tasks(fname, start=0, n=5):
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # split file by '@' separator
    lines = [line.strip() for line in (''.join(lines)).split('@')] 
    common = eval(lines[0]) # get common parameters dict from first line
    lines = lines[1:]
    tasks=[]
    values=[]
    for line in lines:
        text, _, sd = line.partition('#')
        tasks.append(text)
        d = eval(sd) # yes, this is not a good idea, but it works
        values.append(d)
    return [substitute(i, tasks, values, common) for i in range(start, start+n)]


# --- main ---
if __name__ == '__main__':
    tasks = gen_tasks('tasks_pattern_ex.txt', 10)
    for task in tasks:
        print(task)