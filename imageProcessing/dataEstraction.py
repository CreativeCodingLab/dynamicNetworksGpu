from PIL import Image
import numpy as np

#==============================================================================
# 
#==============================================================================

DIRECTORY = 'GIF_resting_state_results/'
DIRECTORY_RESULTS = 'data_extracted/'
N_FRAME = 30
TIME_INSTANTS = 250

#==============================================================================
# 
#==============================================================================

top_left = [85, 30]
top_right = [164, 30]
bottom_left = [85, 109]
bottom_right = [164, 109]

#==============================================================================
# 
#==============================================================================

for f in range(0, (N_FRAME +1)):
    
    print f    
    
    output_info_file_name = 'info_frame_' + str(f) + '.txt'
    output_values_file_name = 'data_frame_' + str(f) + '.txt'
    
    # Open values name
    f_values = open(DIRECTORY_RESULTS + output_values_file_name, 'w')
    
    info = False

    for ts in range(0, (TIME_INSTANTS + 1)):
        
        image_folder = 'time_' + str(ts) + '/'
        image_name = 'frame_' + str(f) + '_time_' + str(ts) + '.png'
        
        # Open the image
        image = Image.open(DIRECTORY + image_folder + image_name)
        
        image_width = image.size[0]
        image_height = image.size[1]
        
        image_gray_values = np.array(image)
        
        # Info
        if not info:
            
            f_info = open(DIRECTORY_RESULTS + output_info_file_name, 'w')
            
            # Write the information
            f_info.write("nodes," + str((top_right[0] - top_left[0]) * (bottom_left[1] - top_left[1]))  + "\n")
            f_info.write("images," + str(TIME_INSTANTS))
            
            f_info.close()
        
        # Write the values of the images
        f_values.write(str(ts))
        
        for r in range(0, image_height):
            for c in range(0, image_width):
                
                if (c >= top_left[0]) and (c <= top_right[0]) and (r >= top_left[1]) and (r <= bottom_left[1]):
                    f_values.write("," + str(image_gray_values[r][c]))
        
        f_values.write(",#\n")
    
    # Close values file
    f_values.close()


