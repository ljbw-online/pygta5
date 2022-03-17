import numpy as np

from common import OUTPUT_SHAPE, oh4_w, oh4_wa, oh4_wd, oh4_s

ae = np.array_equal

correction_data = np.load('correction_data.npy')
f = np.zeros_like(correction_data)
l = f.copy()
r = f.copy()
b = f.copy()

f_count = 0
l_count = 0
r_count = 0
b_count = 0

bd_count = 0

for datum in correction_data:
    output = datum[-1, :OUTPUT_SHAPE]
    if ae(output, oh4_w):
        f[f_count] = datum
        f_count += 1
    elif ae(output, oh4_wa):
        l[l_count] = datum
        l_count += 1
    elif ae(output, oh4_wd):
        r[r_count] = datum
        r_count += 1
    elif ae(output, oh4_s):
        b[b_count] = datum
        b_count += 1
    else:
        print(output)
        exit()

counts = [f_count, l_count, r_count, b_count]
min_count = min(counts)

f = f[:min_count]
l = l[:min_count]
r = r[:min_count]
b = b[:min_count]

print('{} + {} + {} + {} = {}'.format(f_count, l_count, r_count, b_count, sum(counts)))
print('len(correction_data)', len(correction_data))

np.save('unbalanced_correction_data.npy', correction_data)  # Don't overwrite original correction data

correction_data = np.concatenate((f, l, r, b))
print(min_count)
print('len(correction_data)', len(correction_data))
np.save('correction_data.npy', correction_data)
