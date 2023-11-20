import cmath
import math
import numpy as np
import sys
import cv2
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow

def CreateComplexMatrix(dim1: int, dim2:int):
  Z = np.zeros([dim1, dim2], dtype=complex)
  for i in range(dim1):
    for j in range(dim2):
      Z[i,j] = complex(i,j)
  return Z

def FindMinAndMaxComplexMatrix(W):
  min_real = sys.maxsize
  min_imag = sys.maxsize
  max_real = 0
  max_imag = 0
  matrix_size = W.shape

  for i in range(matrix_size[0]):
    for j in range(matrix_size[1]):

      temp_real = W[i,j].real
      temp_imag = W[i,j].imag

      if temp_real >= max_real:
        max_real = temp_real
      if temp_imag >= max_imag:
        max_imag = temp_imag
      if temp_real <= min_real:
        min_real = temp_real
      if temp_imag <= min_imag:
        min_imag = temp_imag

  return [min_real, min_imag, max_real, max_imag]


def Create3DMatrixOfNegativeValue(dim1 : int, dim2: int, value: int):
  M = np.zeros([dim1, dim2, 3]) -value
  return M


def MoebiusValue(current_z, z_p, w_p, denominator_z):
    ##  w - w1     w2 - w3     z - z1     z2 - z3
    ## -------- * --------- = -------- * ---------
    ##  w - w3     w2 - w1     z - z3     z2 - z1

    ##  w - w1     w2 - w1     z - z1     z2 - z3
    ## -------- = --------- * -------- * ---------
    ##  w - w3     w2 - w3     z - z3     z2 - z1


    ##         /  /  w2 - w1     z - z1     z2 - z3  \             \
    ##  w   = |  | --------- * -------- * ---------   | * (w - w3)  | + w1
    ##         \  \  w2 - w3     z - z3     z2 - z1  /             /


    ##    /    /  w2 - w1     z - z1     z2 - z3  \\           /     /  w2 - w1     z - z1     z2 - z3  \ \
    ##    |1 - |  --------- * -------- * --------- || * w   = | -w3*|  --------- * -------- * ---------  | | + w1
    ##    \    \  w2 - w3     z - z3     z2 - z1  //           \     \  w2 - w3     z - z3     z2 - z1  / /


    ##         / /     /  w2 - w1     z - z1     z2 - z3  \ \       \    / /    /  w2 - w1     z - z1     z2 - z3  \\
    ##   w   = || -w3*|  --------- * -------- * ---------  | | + w1  |  /  |1 - |  --------- * -------- * --------- ||
    ##         \ \     \  w2 - w3     z - z3     z2 - z1  / /       /  /   \    \  w2 - w3     z - z3     z2 - z1  //

    value = (-w_p[2]*(((w_p[1] - w_p[0])/(w_p[1] - w_p[2]))*((current_z - z_p[0])/(denominator_z))*((z_p[1] - z_p[2])/(z_p[1] - z_p[0]))) + w_p[0])/(1 - (((w_p[1] - w_p[0])/(w_p[1] - w_p[2]))*((current_z - z_p[0])/(denominator_z))*((z_p[1] - z_p[2])/(z_p[1] - z_p[0]))))
    return value


# Function that generates the complex matrix resulting from the Moebius Transform
# having as parameters Z, a complex matrix, initial and final point of the transform
# respectively z_p, w_p and the type of transform
# (0 = no identiy transform; 1 = identity from left to right;
#  2 = identity from right to left; 3 = identity from bottom to top;
#  other_values = identity from top to bottom) since one could perform
# this transform from different sides of an image
def MoebiusTransform(Z, z_p, w_p, type_tr):
  W = CreateComplexMatrix(Z.shape[0], Z.shape[1])
  denominator_z = complex(1,1)

  min_real, min_imag, max_real, max_imag = FindMinAndMaxComplexMatrix(Z)

  for k in range(W.shape[0]):
    for l in range(W.shape[1]):
      tau_img = Z[k,l].imag / max_imag
      tau_real = Z[k,l].real / max_real

      if Z[k,l] != z_p[2]:
        denominator_z = Z[k,l] - z_p[2]
      else:
        denominator_z = complex(1,1)

      if type_tr == 0:
        W[k,l] = MoebiusValue(Z[k,l], z_p, w_p, denominator_z)
      elif type_tr == 1: #identity from left to right
        W[k,l] = (1 - tau_img)*Z[k,l] + tau_img*MoebiusValue(Z[k,l], z_p, w_p, denominator_z)
      elif type_tr == 2: #identity from right to left
        W[k,l] = tau_img*Z[k,l] + (1 - tau_img)*MoebiusValue(Z[k,l], z_p, w_p, denominator_z)
      elif type_tr == 3: #identity from bottom to top
        W[k,l] = tau_real*Z[k,l] + (1 - tau_real)*MoebiusValue(Z[k,l], z_p, w_p, denominator_z)
      else: #identity from top to bottom
        W[k,l] = (1 - tau_real)*Z[k,l] + tau_real*MoebiusValue(Z[k,l], z_p, w_p, denominator_z)

  return W


## Create a matrix that starting from the Moebius matrix computed earlier subsitutes
## BGR values from the input image into the correct locations, and leaves the others
## with a negative value of -1, that will be used to put those values to black
def ImageFromPixelSubstOfMoebiusTr(image_data, W, dims, max_real, max_imag, value):
  New_Image = Create3DMatrixOfNegativeValue(max_real+1, max_imag+1, value)

  for k in range(dims[0]):
    for g in range(dims[1]):
      if not math.isnan(W[k,g]):
        New_Image[int(W[k,g].real), int(W[k,g].imag), 0] = image_data[k,g, 0]
        New_Image[int(W[k,g].real), int(W[k,g].imag), 1] = image_data[k,g, 1]
        New_Image[int(W[k,g].real), int(W[k,g].imag), 2] = image_data[k,g, 2]

  return New_Image


## OpenCV opens images in BGR, not RGB by default
## Bilinear Interpolation checks all the 8 surrounding pixels at each point
def Bilinear_Interpolation(input_image):
  New_Image = input_image

  for i in range(input_image.shape[0]):
    for j in range(input_image.shape[1]):
      if New_Image[i,j, 0] < 0:
        ## ordered: up, down, left, right, up_left, up_right, down_left, down_right
        i_coords = [input_image.shape[0]-1 if (i-1) < 0 else i-1,
                    0 if (i+1) >= input_image.shape[0] else i+1,
                    i,
                    i,
                    input_image.shape[0]-1 if (i-1) < 0 else i-1,
                    input_image.shape[0]-1 if (i-1) < 0 else i-1,
                    0 if (i+1) >= input_image.shape[0] else i+1,
                    0 if (i+1) >= input_image.shape[0] else i+1]
        j_coords = [j,
                    j,
                    input_image.shape[1]-1 if (j-1) < 0 else j-1,
                    0 if (j+1) >= input_image.shape[1] else j+1,
                    input_image.shape[1]-1 if (j-1) < 0 else j-1,
                    0 if (j+1) >= input_image.shape[1] else j+1,
                    input_image.shape[1]-1 if (j-1) < 0 else j-1,
                    0 if (j+1) >= input_image.shape[1] else j+1]
                    
        up_red = New_Image[i_coords[0], j_coords[0], 2]
        up_green = New_Image[i_coords[0], j_coords[0], 1]
        up_blue = New_Image[i_coords[0], j_coords[0], 0]

        down_red = New_Image[i_coords[1], j_coords[1], 2]
        down_green = New_Image[i_coords[1], j_coords[1], 1]
        down_blue = New_Image[i_coords[1], j_coords[1], 0]

        left_red = New_Image[i_coords[2], j_coords[2], 2]
        left_green = New_Image[i_coords[2], j_coords[2], 1]
        left_blue = New_Image[i_coords[2], j_coords[2], 0]

        right_red = New_Image[i_coords[3], j_coords[3], 2]
        right_green = New_Image[i_coords[3], j_coords[3], 1]
        right_blue = New_Image[i_coords[3], j_coords[3], 0]

        up_left_red = New_Image[i_coords[4], j_coords[4], 2]
        up_left_green = New_Image[i_coords[4], j_coords[4], 1]
        up_left_blue = New_Image[i_coords[4], j_coords[4], 0]

        up_right_red = New_Image[i_coords[5], j_coords[5], 2]
        up_right_green = New_Image[i_coords[5], j_coords[5], 1]
        up_right_blue = New_Image[i_coords[5], j_coords[5], 0]

        down_left_red = New_Image[i_coords[6], j_coords[6], 2]
        down_left_green = New_Image[i_coords[6], j_coords[6], 1]
        down_left_blue = New_Image[i_coords[6], j_coords[6], 0]

        down_right_red = New_Image[i_coords[7], j_coords[7], 2]
        down_right_green = New_Image[i_coords[7], j_coords[7], 1]
        down_right_blue = New_Image[i_coords[7], j_coords[7], 0]

        coords_value_red = [up_red, down_red, left_red, right_red, up_left_red, up_right_red, down_left_red, down_right_red]
        coords_value_green = [up_green, down_green, left_green, right_green, up_left_green, up_right_green, down_left_green, down_right_green]
        coords_value_blue = [up_blue, down_blue, left_blue, right_blue, up_left_blue, up_right_blue, down_left_blue, down_right_blue]

        count = 0
        value_red = 0
        value_green = 0
        value_blue = 0

        for m in range(len(coords_value_red)):
          if coords_value_red[m] >= 0:
            count = count + 1
            value_red = value_red + coords_value_red[m]
            value_green = value_green + coords_value_green[m]
            value_blue = value_blue + coords_value_blue[m]

        if count != 0:
          New_Image[i,j,2] = value_red/count
          New_Image[i,j,1] = value_green/count
          New_Image[i,j,0] = value_blue/count
        else:
          New_Image[i,j,2] = 0
          New_Image[i,j,1] = 0
          New_Image[i,j,0] = 0

  return New_Image

if __name__ == "__main__":
  # Load and show input image
  img_color = cv2.imread('building.jpg',1)
  cv2_imshow(img_color)
  #plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
  #plt.axis("off")
  #plt.show()
  dims = img_color.shape

  Z = CreateComplexMatrix(dims[0], dims[1])
  zp = [Z[1016, 47], Z[447, 277], Z[141, 564]]
  wp = [Z[1016, 47], Z[447, 47], Z[141, 47]]

  W = MoebiusTransform(Z, zp, wp, 0)

  min_real, min_imag, max_real, max_imag = FindMinAndMaxComplexMatrix(W)

  ## adjust values
  for i in range(W.shape[0]):
    for j in range(W.shape[1]):
      W[i,j] = complex(W[i,j].real - min_real, W[i,j].imag - min_imag)

  max_imag_ = max_imag - min_imag + 1
  max_real_ = max_real - min_real + 1

  New_Image_building = Bilinear_Interpolation(ImageFromPixelSubstOfMoebiusTr(img_color, W, dims, int(max_real_), int(max_imag_), 1))

  #cv2.imwrite('moebius.jpg', New_Image_building)
  #img_color_moebius = cv2.imread('moebius.jpg',1)
  #plt.imshow(cv2.cvtColor(img_color_moebius, cv2.COLOR_BGR2RGB))
  #plt.axis("off")
  #plt.show()

  cv2_imshow(New_Image_building)
