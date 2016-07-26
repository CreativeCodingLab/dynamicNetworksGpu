from PIL import Image
import os

#==============================================================================
# 
#==============================================================================

DIRECTORY = 'GIF_resting_state/'
DIRECTORY_SAVE = 'GIF_resting_state_results/'
N_FRAME = 30

#==============================================================================
# 
#==============================================================================

for img_gif in os.listdir(DIRECTORY):
    
    print img_gif
    aux = img_gif
    aux = aux.replace('MRI_GIF','')
    time = aux.replace('.gif', '')
    
    
    im = Image.open(DIRECTORY + img_gif)
    
    DIRECTORY_FRAME = DIRECTORY_SAVE + 'time_' + str(time) + '/'
    if not os.path.exists(DIRECTORY_FRAME):
        os.makedirs(DIRECTORY_FRAME)    
    
    for i in range(0,N_FRAME):
#        print i
        im.seek(im.tell()+1)
        im.save(DIRECTORY_FRAME + 'frame_' + str(i) + '_time_' + str(time) + '.png')
    


