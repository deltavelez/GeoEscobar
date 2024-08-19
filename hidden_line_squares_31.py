#C:\Users\Diego\AppData\Local\Programs\Python\Python311\Lib\site-packages\pygame\examples
# https://www.geeksforgeeks.org/sorting-objects-of-user-defined-class-in-python/
from dataclasses import dataclass
import numpy as np
import pygame
from pygame.locals import *
import time
import sys
import random
import multiprocessing

# inicializacion de la libreria PyGame, de la ventana, y de la fuente
# Definicion del tamano de la ventana
screen_width = 1080
screen_height = 1080
pygame.init()
pygame.display.set_caption('Graphics World')
pygame.font.init()
my_font = pygame.font.SysFont('arial', 30)

def Zero(x0, y0, x1, y1):
    return x0-y0*(x1-x0)/(y1-y0)

def FindZero(x0, y0, x1, y1):
    if y0<0 and y1<0:
        return False, 0
    if y0>0 and y1>0:
        return False, 0
    if y0==0 and y1==0:
        return True, 0.5*(x0+x1)
    if y0==y1:
        return False, 0
    return True, x0-y0*(x1-x0)/(y1-y0)

def FindZeroVec(A, za, B, zb):
    if za==0 and zb==0:
        return True, A + 0.5*(B-A)
    if za==zb:
        return False, [0,0,0] 
    return za*zb<=0,  A-(B-A)*za/(zb-za)

PLANE_BORDER_WIDTH_MASK         = 0b00000011
PLANE_LIGHT_EFFECT              = 0b00000100
PLANE_LIGHT_EFFECT_OFFSET       = 2
PLANE_FILL                      = 0b00001000
PLANE_FILL_OFFSET               = 3

A = np.array([-1,0,0])
B = np.array([ 1,0,0])
C = np.array([ 0,1,0])
P = np.array([ 0.1,0.7,0])

def bari_to_cartesian(x1, y1, x2, y2, x3, y3, L1, L2, L3):
    x = L1*x1 + L2*x2 + L3*x3
    y = L1*y1 + L2*y2 + L3*y3
    return x, y

def cartesian_to_bari(x1, y1, x2, y2, x3, y3, x, y):
    temp = (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
    L1 = 0.0
    L2 = 0.0
    L3 = 0.0
    
    if temp!=0:
        L1 = ((y2-y3)*(x-x3)+(x3-x2)*(y-y3))/temp
    else:
        L1 = 0
    temp = (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
    if temp!=0:
        L2 = ((y3-y1)*(x-x3)+(x1-x3)*(y-y3))/temp
    else:
        L2 = 0.
    return L1, L2, 1.0-L1-L2

w0, w1, w2 = cartesian_to_bari(-1, 0, 1, 0, 0, 1, 0.25,.33)
#print("lambda =",w0, w1, w2)
x, y = bari_to_cartesian(-1, 0, 1, 0, 0, 1, w0, w1, w2)
#print("P=",x,y)

##def bari_to_cartesian(A, B, C, W):
##    return W[0]*A + W[1]*B + W[2]*C;
##
##def cartesian_to_bari(A, B, C, P):
##    V0 = B - A
##    V1 = C - A
##    V2 = P - A;
##    d00 = np.dot(V0,V0);
##    d01 = np.dot(V0,V1);
##    d11 = np.dot(V1,V1);
##    d20 = np.dot(V2,V0);
##    d21 = np.dot(V2,V1);
##    temp = d00 * d11 - d01 * d01;
##    if temp!=0:
##        temp = 1/temp
##        w1 = temp*(d11 * d20 - d01 * d21);
##        w2 = temp*(d00 * d21 - d01 * d20);
##        w0 = 1.0 - w1 - w2;
##    else:
##        w0 = 0
##        w1 = 0
##        w2 = 0
##    return np.array([w0, w1, w2])

    
##print("P =",P)
##w = cartesian_to_bari(A, B, C, P)
##print("lambda =",w)
##p = bari_to_cartesian(A, B, C, w)
##print("P=",p)
##print("delta=",P-p)
    
@dataclass
class PlaneObject:
    geo: list
    flags: bool
    color_border: tuple
    color: tuple
    distance : float
    def __repr__(self): 
        return str((self.geo, self.flags, self.color, self.distance))

class VP2d:
    xp_min: float
    xp_max: float
    yp_min: float
    yp_max: float
    xr_min: float
    xr_max: float
    yr_min: float
    yr_max: float
    
def project_2d(vp2d, R):
    P = [ 0.0, 0.0 ]
    P[0]= vp2d.xp_min + (R[0]-vp2d.xr_min)*(vp2d.xp_max-vp2d.xp_min)/(vp2d.xr_max-vp2d.xr_min)
    P[1]= vp2d.yp_max + (R[1]-vp2d.yr_min)*(vp2d.yp_min-vp2d.yp_max)/(vp2d.yr_max-vp2d.yr_min)
    return P
 
def draw_line_2d(surface, vp2d, R0, R1, w, color):
    P0 = project_2d(vp2d,R0)
    P1 = project_2d(vp2d,R1)
    pygame.draw.line(surface, color, P0, P1, w)



def draw_vector(surface,P0, P1, length, angle, width, color):
    pygame.draw.line(surface, color, P0, P1, width)
    if length==0  or (P0[0] == P1[0] and P0[1]==P1[1]):
        return;
    t = 1.0 - length/np.sqrt((P0[0]-P1[0])**2 + (P0[1]-P1[1])**2)
    lx = P1[0] - P0[0] - t*(P1[0]-P0[0])
    ly = P1[1] - P0[1] - t*(P1[1]-P0[1])
    c = np.cos(angle)
    s = np.sin(angle)
    lxp = c*lx - s*ly
    lyp = s*lx + c*ly
    pygame.draw.line(surface, color, P1, [P1[0]-lxp, P1[1]-lyp], width)
    lxp =  c*lx + s*ly
    lyp = -s*lx + c*ly
    pygame.draw.line(surface, color, P1, [P1[0]-lxp, P1[1]-lyp], width)

def draw_vector_2d(surface, vp2d, R0, R1, length, angle, width, color):
    P0 = project_2d(vp2d,R0)
    P1 = project_2d(vp2d,R1)
    draw_vector(surface, [P0[0], P0[1]], [P1[0], P1[1]],  length, angle, width, color)

def draw_rectangle_2d(surface, vp2d, R0, R1, w, color):
    P0 = project_2d(vp2d,R0)
    P1 = project_2d(vp2d,R1)
    pygame.draw.line(surface, color, P0, [P1[0],P0[1]], w)
    pygame.draw.line(surface, color, [P1[0],P0[1]], P1, w)
    pygame.draw.line(surface, color, P1, [P0[0],P1[1]], w)
    pygame.draw.line(surface, color, [P0[0],P1[1]], P0, w)

class MathImplicitCurves:
    def __init__(self, size_x, size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.invalid = np.zeros((size_x, size_y), dtype='bool')
        self.R =  np.zeros((self.size_x, self.size_y))
        self.Fi = np.zeros((self.size_x-1, self.size_y), dtype='bool')
        self.Fj = np.zeros((self.size_x, self.size_y-1), dtype='bool')
        self.Zi = np.zeros((self.size_x-1, self.size_y))
        self.Zj = np.zeros((self.size_x, self.size_y-1))
                
    def CalcScalarField(self, f, xa, xb, ya, yb):
        self.xa = xa
        self.xb = xb
        self.ya = ya
        self.yb = yb
        self.dx = (xb-xa)/(self.size_x-1)
        self.dy = (yb-ya)/(self.size_y-1)
        y = ya
        for j in range(self.size_y):
            x = xa
            for i in range(self.size_x):
                self.R[i,j], self.invalid[i,j] = f(x,y)
                x = x + self.dx
            y = y + self.dy
        
    def FindZeros(self, vp2d):
        y = self.ya 
        for j in range(self.size_y):
            x = self.xa
            for i in range(self.size_x):
                Point = project_2d(vp2d, [x,y])
                # pygame.draw.circle(surface, (0,0,255), Point, 5)
                if i<self.size_x-1:
                    self.Fi[i,j], self.Zi[i,j] = FindZero(x,self.R[i,j],x+self.dx,self.R[i+1,j])
                    #if self.Fi[i,j]:
                    #    pygame.draw.circle(surface, (255,0,0), project_2d(vp2d, [self.Zi[i,j],y]), 5)
                     
                if j<self.size_y-1:
                    self.Fj[i,j], self.Zj[i,j] = FindZero(y,self.R[i,j],y+self.dy,self.R[i,j+1])
                    #if self.Fj[i,j]:
                    #   pygame.draw.circle(surface, (255,0,0), project_2d(vp2d, [x, self.Zj[i,j]]), 5)

                x = x + self.dx
            y = y + self.dy

    def WalkingSquare(self, surface, vp2d, z, width, color):
        y = self.ya 
        for j in range(self.size_y-1):
            x = self.xa
            for i in range(self.size_x-1):
                s0 = (self.R[i,j]-z > 0)
                s1 = (self.R[i+1,j]-z > 0)
                s2 = (self.R[i+1,j+1]-z >0)
                s3 = (self.R[i,j+1]-z >0)
                #pygame.draw.circle(surface, (0,0,255), project_2d(vp2d, P01), 2)
                if s0!=s1:
                    P01 = [Zero(x,self.R[i,j]-z,x+self.dx,self.R[i+1,j]-z),y]
#                    pygame.draw.circle(surface, (255,0,0), project_2d(vp2d, P01), 5)
               
                if s1!=s2:
                    P12 = [x+self.dx,Zero(y,self.R[i+1,j]-z,y+self.dy,self.R[i+1,j+1]-z)]
#                    pygame.draw.circle(surface, (255,255,0), project_2d(vp2d, P12), 5)
                if s2!=s3:
                    P23 = [Zero(x,self.R[i,j+1]-z,x+self.dx,self.R[i+1,j+1]-z),y+self.dy]
#                    pygame.draw.circle(surface, (127,0,0), project_2d(vp2d, P23), 5)

                if s0!=s3:
                    P03 = [x,Zero(y,self.R[i,j]-z,y+self.dy,self.R[i,j+1]-z)]
 #                   pygame.draw.circle(surface, (127,0,127), project_2d(vp2d, P03), 5)

                # Case 2
                if not(s3) and not(s2) and not(s1) and s0:
                    draw_line_2d(surface, vp2d, P03, P01, width, color)

                # Case 3
                if not(s3) and not(s2) and s1 and not(s0):
                    draw_line_2d(surface, vp2d, P01, P12, width, color)

                # Case 4
                if not(s3) and not(s2) and s1 and s0:
                    draw_line_2d(surface, vp2d, P03, P12, width, color)
                    
                # Case 5
                if not(s3) and s2 and not(s1) and not(s0):
                    draw_line_2d(surface, vp2d, P12, P23, width, color)

                # Case 6
                if not(s3) and s2 and not(s1) and s0:
                    draw_line_2d(surface, vp2d, P03, P23, width, color)
                    draw_line_2d(surface, vp2d, P01, P12, width, color)

                # Case 7
                if not(s3) and s2 and s1 and not(s0):
                    draw_line_2d(surface, vp2d, P01, P23, width, color)

                # Case 8
                if not(s3) and s2 and s1 and s0:
                    draw_line_2d(surface, vp2d, P03, P23, width, color)

                # Case 9
                if s3 and not(s2) and not(s1) and not(s0):
                    draw_line_2d(surface, vp2d, P03, P23, width, color)

                # Case 10
                if s3 and not(s2) and not(s1) and s0:
                    draw_line_2d(surface, vp2d, P01, P23, width, color)

                # Case 11
                if s3 and not(s2) and s1 and not(s0):
                    draw_line_2d(surface, vp2d, P01, P03, width, color)
                    draw_line_2d(surface, vp2d, P12, P23, width, color)

                # Case 12
                if s3 and not(s2) and s1 and s0:
                    draw_line_2d(surface, vp2d, P12, P23, width, color)

                # Case 13
                if s3 and s2 and not(s1) and not(s0):
                    draw_line_2d(surface, vp2d, P03, P12, width, color)

                # Case 14
                if s3 and s2 and not(s1) and s0:
                    draw_line_2d(surface, vp2d, P01, P12, width, color)

                # Case 15
                if s3 and s2 and s1 and not(s0):
                    draw_line_2d(surface, vp2d, P01, P03, width, color)
                    
                x = x + self.dx
            y = y + self.dy

class VP3d:
    o_x: int       # coordenada x del centro de proyeccion
    o_y: int       # coordenada y del centro de proyeccion
    scale: float   # parametro de escala
    cam: np.ndarray  # coordenadas x,y,z de la camara
    cam_d: float   # inter-eye distance for stereoscopic projection
    cam_h: float   # lens-to-retina distance
    iso:bool       # isometric or perspective flag
    light:  np.ndarray          # coordenadas x,y,z de la fuente de luz

def transform_3d(vp3d, Rot, P):
    T = np.matmul(Rot,P)
    T = T - vp3d.cam
    return T

def project_3d(vp3d, P):
    pp0 = 0.0
    pp1 = 0.0
    
    if vp3d.iso:
        pp0 = P[0]
        pp1 = P[1]
    else:
        pp0 = vp3d.cam_h*P[0] / (vp3d.cam_h-P[2])
        pp1 = vp3d.cam_h*P[1] / (vp3d.cam_h-P[2])
        
    xp = vp3d.o_x + vp3d.scale*pp0
    yp = vp3d.o_y - vp3d.scale*pp1
    return [xp, yp, P[2]]

def plot_3d(vp3d, Rot, P):
    T = transform_3d(vp3d, Rot, P)
    return project_3d(vp3d,T)

def plot_4d_2d(xc, yc, scale, Rot, P):
    T = np.matmul(Rot,P)

    if True:
        xp = xc + scale*T[0]
        yp = yc - scale*T[1]
    else:
        a= 3
        xp = xc + scale*T[0]*a/(a+T[3])
        yp = yc - scale*T[1]*a/(a+T[3])
    return [xp,yp]


def draw_line_3d(surface, vp3d, Rot, R0, R1, w, color):
    P0 = plot_3d(vp3d, Rot, R0)
    P1 = plot_3d(vp3d, Rot, R1)
    pygame.draw.line(surface, color, [P0[0], P0[1]], [P1[0], P1[1]], w)


def plot_dot(surface, vp3d, Rot, P, radius, color):
    T = transform_3d(vp3d, Rot, P)
    P = project_3d(vp3d,T)
    pygame.draw.circle(surface, color, [P[0],P[1]], radius) 

def create_rot_x(tetha):
    c = np.cos(tetha)
    s = np.sin(tetha)
    Rx = np.zeros((3, 3))
    Rx[0][0]=1; Rx[0][1]=0; Rx[0][2]= 0
    Rx[1][0]=0; Rx[1][1]=c; Rx[1][2]=-s  
    Rx[2][0]=0; Rx[2][1]=s; Rx[2][2]= c
    return Rx

def create_rot_y(tetha):
    c = np.cos(tetha)
    s = np.sin(tetha)
    Ry = np.zeros((3, 3))
    Ry[0][0]= c; Ry[0][1]=0; Ry[0][2]=s
    Ry[1][0]= 0; Ry[1][1]=1; Ry[1][2]=0  
    Ry[2][0]=-s; Ry[2][1]=0; Ry[2][2]=c
    return Ry

def create_rot_z(tetha):
    c = np.cos(tetha)
    s = np.sin(tetha)
    Rz = np.zeros((3, 3))
    Rz[0][0]=c; Rz[0][1]=-s; Rz[0][2]=0
    Rz[1][0]=s; Rz[1][1]= c; Rz[1][2]=0  
    Rz[2][0]=0; Rz[2][1]= 0; Rz[2][2]=1
    return Rz

# four-dimensional rotations
def create_rot_xy(tetha):
    c = np.cos(tetha)
    s = np.sin(tetha)
    Rxy = np.zeros((4, 4))
    Rxy[0][0]=c; Rxy[0][1]=-s; Rxy[0][2]=0; Rxy[0][3]=0
    Rxy[1][0]=s; Rxy[1][1]= c; Rxy[1][2]=0; Rxy[1][3]=0  
    Rxy[2][0]=0; Rxy[2][1]= 0; Rxy[2][2]=1; Rxy[2][3]=0
    Rxy[3][0]=0; Rxy[3][1]= 0; Rxy[3][2]=0; Rxy[3][3]=1
    return Rxy

def create_rot_xz(tetha):
    c = np.cos(tetha)
    s = np.sin(tetha)
    Rxz = np.zeros((4, 4))
    Rxz[0][0]=c; Rxz[0][1]= 0; Rxz[0][2]=-s; Rxz[0][3]=0
    Rxz[1][0]=0; Rxz[1][1]= 1; Rxz[1][2]= 0; Rxz[1][3]=0  
    Rxz[2][0]=s; Rxz[2][1]= 0; Rxz[2][2]= c; Rxz[2][3]=0
    Rxz[3][0]=0; Rxz[3][1]= 0; Rxz[3][2]= 0; Rxz[3][3]=1
    return Rxz

def create_rot_xw(tetha):
    c = np.cos(tetha)
    s = np.sin(tetha)
    Rxw = np.zeros((4, 4))
    Rxw[0][0]=c; Rxw[0][1]= 0; Rxw[0][2]= 0; Rxw[0][3]=-s
    Rxw[1][0]=0; Rxw[1][1]= 1; Rxw[1][2]= 0; Rxw[1][3]=0  
    Rxw[2][0]=0; Rxw[2][1]= 0; Rxw[2][2]= 1; Rxw[2][3]=0
    Rxw[3][0]=s; Rxw[3][1]= 0; Rxw[3][2]= 0; Rxw[3][3]=c
    return Rxw

def create_rot_yz(tetha):
    c = np.cos(tetha)
    s = np.sin(tetha)
    Ryz = np.zeros((4, 4))
    Ryz[0][0]=1; Ryz[0][1]= 0; Ryz[0][2]= 0; Ryz[0][3]=0
    Ryz[1][0]=0; Ryz[1][1]= c; Ryz[1][2]=-s; Ryz[1][3]=0  
    Ryz[2][0]=0; Ryz[2][1]= s; Ryz[2][2]= c; Ryz[2][3]=0
    Ryz[3][0]=s; Ryz[3][1]= 0; Ryz[3][2]= 0; Ryz[3][3]=1
    return Ryz

def create_rot_yw(tetha):
    c = np.cos(tetha)
    s = np.sin(tetha)
    Ryw = np.zeros((4, 4))
    Ryw[0][0]=1; Ryw[0][1]= 0; Ryw[0][2]= 0; Ryw[0][3]= 0
    Ryw[1][0]=0; Ryw[1][1]= c; Ryw[1][2]= 0; Ryw[1][3]=-s  
    Ryw[2][0]=0; Ryw[2][1]= 0; Ryw[2][2]= 1; Ryw[2][3]= 0
    Ryw[3][0]=0; Ryw[3][1]= s; Ryw[3][2]= 0; Ryw[3][3]= c
    return Ryw

def create_rot_zw(tetha):
    c = np.cos(tetha)
    s = np.sin(tetha)
    Rzw = np.zeros((4, 4))
    Rzw[0][0]=1; Rzw[0][1]= 0; Rzw[0][2]= 0; Rzw[0][3]= 0
    Rzw[1][0]=0; Rzw[1][1]= 1; Rzw[1][2]= 0; Rzw[1][3]= 0  
    Rzw[2][0]=0; Rzw[2][1]= 0; Rzw[2][2]= c; Rzw[2][3]=-s
    Rzw[3][0]=0; Rzw[3][1]= 0; Rzw[3][2]= s; Rzw[3][3]= c
    return Rzw


def create_tait_bryan_ZYX(yaw, pitch, roll):
    Rx = create_rot_x(roll)
    Ry = create_rot_y(pitch)
    Rz = create_rot_z(yaw)
    return np.matmul(Rz, np.matmul(Ry, Rx))

def get_tait_bryan_ZYX(R):
    # prevent numerical stability issue with degenerated rotation matrices
    temp = R[2,0]
    
    if temp<-1:
        temp=-1
    if temp>1:
        temp=1
    pitch = np.arcsin(-temp)
    yaw = np.arctan2(R[1,0],R[0,0])
    roll = np.arctan2(R[2,1],R[2,2])
    return yaw, pitch, roll


def draw_vector_3d(surface, A, B, vp3d, Rot, length, angle, width, color):
    PA = plot_3d(vp3d, Rot, A)
    PB = plot_3d(vp3d, Rot, B)
    draw_vector(surface, [PA[0],PA[1]], [PB[0],PB[1]], length, angle, width, color)

def draw_box_3d(surface, A,B, vp3d, Rot, width, color):
    P0 = plot_3d(vp3d, Rot, A)
    P1 = plot_3d(vp3d, Rot, [B[0],A[1],A[2]])
    P2 = plot_3d(vp3d, Rot, [A[0],B[1],A[2]])
    P3 = plot_3d(vp3d, Rot, [B[0],B[1],A[2]])
    P4 = plot_3d(vp3d, Rot, [A[0],A[1],B[2]])
    P5 = plot_3d(vp3d, Rot, [B[0],A[1],B[2]])
    P6 = plot_3d(vp3d, Rot, [A[0],B[1],B[2]])
    P7 = plot_3d(vp3d, Rot, B)
    
    pygame.draw.line(surface, color, [P0[0], P0[1]], [P1[0], P1[1]], width)
    pygame.draw.line(surface, color, [P1[0], P1[1]], [P3[0], P3[1]], width)
    pygame.draw.line(surface, color, [P3[0], P3[1]], [P2[0], P2[1]], width)
    pygame.draw.line(surface, color, [P2[0], P2[1]], [P0[0], P0[1]], width)

    pygame.draw.line(surface, color, [P0[0], P0[1]], [P4[0], P4[1]], width)
    pygame.draw.line(surface, color, [P1[0], P1[1]], [P5[0], P5[1]], width)
    pygame.draw.line(surface, color, [P3[0], P3[1]], [P7[0], P7[1]], width)
    pygame.draw.line(surface, color, [P2[0], P2[1]], [P6[0], P6[1]], width)

    pygame.draw.line(surface, color, [P4[0], P4[1]], [P5[0], P5[1]], width)
    pygame.draw.line(surface, color, [P5[0], P5[1]], [P7[0], P7[1]], width)
    pygame.draw.line(surface, color, [P7[0], P7[1]], [P6[0], P6[1]], width)
    pygame.draw.line(surface, color, [P6[0], P6[1]], [P4[0], P4[1]], width)
    
# Reflection of a vector on a plane.  N must be normalized.
def reflection(A, N):
    n_norm2 = np.dot(N, N)
    if n_norm2 != 0:
        return -A + 2*np.dot(A, N)*N/n_norm2
#    print(N)
#    print("Error (reflection)")
    return [0,0,0]

# Funcion para escribir texto en pantalla
def draw_text(surface, x, y, message, color):
    text_surface = my_font.render(message, False, color)
    surface.blit(text_surface, (x,y))

# Constantes para conversion angular entre radianes y grados sexagesimales
D2R = np.pi/180.0;  R2D = 180.0/np.pi

# Matrices para efectuar rotaciones incrementales de la `camara'
# a traves de las teclas de direccion y page_up y page_down
Ym = create_rot_z(-D2R); Yp = create_rot_z( D2R) # Yaw   (minus/plus)
Pm = create_rot_y(-D2R); Pp = create_rot_y( D2R) # Pitch (minus/plus)
Rm = create_rot_x(-D2R); Rp = create_rot_x( D2R) # Roll  (minus/plus)

# Dibuja un avion sencillo 
def Draw_Plane(surface, vp, R):
    a = [[4,0,0], [3,0.5,0], [1,0.5,0], [-1,3,0], [-1,0.5,0], [-3,0.5,0],
        [-3,-0.5,0], [-1,-0.5,0], [-1,-3,0], [1,-0.5,0], [3,-0.5,0], [4,0,0]]

    a = np.asarray(a)
    for k in range(len(a)-1):
        xa, ya, za = project_3d(vp, a[k])
        xb, yb, zb = project_3d(vp, a[k+1])
        if za<0 and zb<0:
            pygame.draw.line(surface, "red", [xa, ya], [xb, yb], 3)

    a = [ [-2,0,0], [-3,0,1], [-3,0,0], [-2,0,0] ]
    a = np.asarray(a)
    for k in range(len(a)-1):
        xa, ya, za = project_3d(vp, a[k])
        xb, yb, zb = project_3d(vp, a[k+1])
        if za<0 and zb<0:
            pygame.draw.line(surface, "blue", [xa, ya], [xb, yb], 3)

# Esta rutina presenta la manera de la cual se miden los vectores H y G
# referidos al sistema coordenado X'Y'Z' de la aeronave.

TOLERANCE = 1E-6
def is_null(P):
    if np.abs(P[0])>TOLERANCE or np.abs(P[1])>TOLERANCE or np.abs(P[2])>TOLERANCE:
        return False
    return True

class MathCurve:
    def __init__(self, n, color, flags):
        self.n = n
        self.R = np.zeros((n,3))
        self.T = np.zeros((n,3))
        self.P = np.zeros((n,3))
        self.invalid = np.zeros(n, dtype='bool')
        self.color = color
        self.flags = flags

    def Calc(self, fn, ta, tb):
        dt = (tb-ta)/(self.n-1)
        t = ta
        for j in range(self.n):
            self.R[j], self.invalid[j] = fn(t)
            t = t + dt
        
    def Transform(self, vp, R):
        self.light_T = np.matmul(R,vp.light)
        self.light_T = self.light_T - vp.cam
        for j in range(self.n):
            self.T[j] = np.matmul(R,self.R[j])
            self.T[j] = self.T[j] - vp.cam

    def Project(self, vp, R):
        for j in range(self.n):
            self.P[j] = project_3d(vp, self.T[j])

    def AddSegments(self, vp, planes):
        for j in range(self.n-1):
            E = np.array([0,0,1])
            if self.P[j,2]<0 and self.P[j+1,2]<0:
                if ((self.flags & PLANE_LIGHT_EFFECT)>>PLANE_LIGHT_EFFECT_OFFSET) == 0x01:
                     
                    W = 0.5*(self.T[j] + self.T[j+1])
                    C = np.cross(E-W,self.light_T-W)
                    norm = np.linalg.norm(E-W)*np.linalg.norm(self.light_T-W)
                    if norm !=0:
                        tetha = np.arccos(np.dot(E-W,self.light_T-W)/norm)
                        r = np.sin(tetha)
                        r = 1-np.sqrt(0.5*(1-r))
                        r = 0.333 + 0.666*r   
                        red,green,blue = self.color
                        
                        red = int(r*red)
                        green = int(r*green)
                        blue = int(r*blue)
                        P = PlaneObject( [[ self.P[j,0], self.P[j,1]], [self.P[j+1,0], self.P[j+1,1]] ],
                                         self.flags, (0,0,0), (red, green, blue), 0.5*(self.P[j,2]+self.P[j+1,2]))
                        planes.append(P)


##    P = PlaneObject([ [self.P[i,j,0], self.P[i,j,1]],
##                                      [self.P[i,j+1,0], self.P[i,j+1,1]],
##                                      [self.P[i+1,j+1,0], self.P[i+1,j+1,1]],
##                                      [self.P[i+1,j,0], self.P[i+1,j,1]] ],
##                                      self.flags, self.color_line, (red,green,blue),
##                                      0.25*(self.P[i,j,2]+self.P[i,j+1,2]+self.P[i+1,j+1,2]+self.P[i+1,j,2]))
                        
    # E: eye
    # R: reflected light
    # N: normal vector to the plane
    # W: middle point in the cell 

class MathSurface:
    def __init__(self, m, n, color_line, color_top, color_bottom, flags):
        self.m = m
        self.n = n
        self.R = np.zeros((m, n,3))
        self.T = np.zeros((m, n,3))
        self.P = np.zeros((m, n,3))
        self.invalid = np.zeros((m, n), dtype='bool')
        self.color_line = color_line
        self.color_top = color_top
        self.color_bottom = color_bottom
        self.flags = flags
        
    def Calc(self, fn, ta, tb, ua, ub):
        du = (ub-ua)/(self.m-1)
        dt = (tb-ta)/(self.n-1)
        u = ua
        for i in range(self.m):
            t = ta
            for j in range(self.n):
                self.R[i,j], self.invalid[i,j] = fn(t,u)
                t = t + dt
            u = u + du

    def CalcAxis(self, P0, P1, radius):
        A = P1-P0
        if is_null(A):
            return
        else:
            if A[0]!=0:
                norm = np.sqrt(((A[1]+A[2])/A[0])**2+2)
                B = np.array([(-A[1]-A[2])/A[0], 1,1])/norm
                C = np.cross(B, A/np.linalg.norm(A))
            elif A[1]!=0:
                norm = np.sqrt(((A[0]+A[2])/A[1])**2+2)
                B = np.array([1, (-A[0]-A[2])/A[1], 1])/norm
                C = np.cross(B, A/np.linalg.norm(A))
            else:
                norm = np.sqrt(((A[1]+A[0])/A[2])**2+2)
                B = np.array([1,1,(-A[1]-A[0])/A[2]])/norm
                C = np.cross(B, A/np.linalg.norm(A))

            B = radius*B
            C = radius*C
        
            du = 1.0/(self.m-1)
            dt = 2*np.pi/(self.n-1)
            u = 0
            for i in range(self.m):
                t = 0
                for j in range(self.n):
                    self.R[i,j] = P0+u*(P1-P0) + B*np.cos(t) + C*np.sin(t)
                    t = t + dt
                u = u + du
            
    def Transform(self, vp, R):
        self.light_T = np.matmul(R,vp.light)
        self.light_T = self.light_T - vp.cam
        for i in range(self.m):
            for j in range(self.n):
                self.T[i,j] = np.matmul(R,self.R[i,j])
                self.T[i,j] = self.T[i,j] - vp.cam

    def Project(self, vp, R):
        for i in range(self.m):
            for j in range(self.n):
                self.P[i,j] = project_3d(vp, self.T[i,j])

    def AddPlanes(self, planes):
        for i in range(self.m-1):
            for j in range(self.n-1):
                if not(self.invalid[i,j] or self.invalid[i,j+1] or self.invalid[i+1,j+1] or self.invalid[i+1,j]) and \
                   self.P[i,j,2]<0 and self.P[i,j+1,2]<0 and self.P[i+1,j+1,2]<0 and self.P[i+1,j,2]<0:
                    E = np.array([0,0,1])
                    
                    # N: normal vector to the plane
                    N = np.cross(self.T[i,j+1]-self.T[i,j],self.T[i+1,j]-self.T[i,j])
                    if is_null(N):
                        N = np.cross(self.T[i+1,j+1]-self.T[i,j+1],self.T[i,j]-self.T[i,j+1])
                    if is_null(N):
                        N = np.cross(self.T[i,j]-self.T[i+1,j],self.T[i+1,j+1]-self.T[i+1,j])
                    if is_null(N):
                        N = np.cross(self.T[i+1,j]-self.T[i+1,j+1],self.T[i,j+1]-self.T[i+1,j+1])
                    if is_null(N):
                        N = np.cross(self.T[i,j+1]-self.T[i,j],self.T[i+1,j+1]-self.T[i,j])
                    if is_null(N):
                        N = np.cross(self.T[i+1,j+1]-self.T[i,j],self.T[i+1,j]-self.T[i,j])
                    if is_null(N):
                        N = np.cross(self.T[i+1,j+1]-self.T[i,j+1],self.T[i+1,j]-self.T[i,j+1])
                    if is_null(N):
                        N = np.cross(self.T[i+1,j]-self.T[i,j+1],self.T[i,j]-self.T[i,j+1])
                    if is_null(N):
                        N = np.cross(self.T[i+1,j]-self.T[i+1,j+1],self.T[i,j]-self.T[i+1,j+1])
                    if is_null(N):
                        N = np.cross(self.T[i,j]-self.T[i+1,j+1],self.T[i,j+1]-self.T[i+1,j+1])
                    if is_null(N):
                        N = np.cross(self.T[i,j+1]-self.T[i+1,j],self.T[i+1,j+1]-self.T[i+1,j])
                    if is_null(N):
                        N = np.cross(self.T[i,j]-self.T[i+1,j],self.T[i,j+1]-self.T[i+1,j])
                    if is_null(N):
                        continue
                    
                    # W: middle point within the cell element
                    W = 0.25*(self.T[i,j]+self.T[i,j+1]+self.T[i+1,j+1]+self.T[i+1,j])
                   
                    if ((self.flags & PLANE_LIGHT_EFFECT)>>PLANE_LIGHT_EFFECT_OFFSET) == 0x01:
                        # R: reflected light
                        R = reflection(self.light_T-W,N)
                        norm = np.linalg.norm(R)*np.linalg.norm(E-W)
                        if norm==0:
                            if np.dot(N,E)>0:
                                red,green,blue = self.color_top
                            else:
                                red,green,blue = self.color_bottom
                        else:
                            r = np.dot(R,E-W)/norm
                            # model 1
                            r = 1-np.sqrt(0.5*(1-r))
                            r = 0.333 + 0.666*r
                            if np.dot(N,E)>0:
                                red,green,blue = self.color_top
                            else:
                                red,green,blue = self.color_bottom
                            red = int(r*red)
                            green = int(r*green)
                            blue = int(r*blue)
                    else:
                        if np.dot(N,E)>0:
                            red,green,blue = self.color_top
                        else:
                            red,green,blue = self.color_bottom

                    P = PlaneObject([ [self.P[i,j,0], self.P[i,j,1]],
                                      [self.P[i,j+1,0], self.P[i,j+1,1]],
                                      [self.P[i+1,j+1,0], self.P[i+1,j+1,1]],
                                      [self.P[i+1,j,0], self.P[i+1,j,1]] ],
                                      self.flags, self.color_line, (red,green,blue),
                                      0.25*(self.P[i,j,2]+self.P[i,j+1,2]+self.P[i+1,j+1,2]+self.P[i+1,j,2]))
                    planes.append(P)
    # E: eye
    # R: reflected light
    # N: normal vector to the plane
    # W: middle point in the cell 

    def AddTriangle(self, ra, ca, rb, cb, P, Pp, vp, planes):
        if self.P[ra,ca,2]<0 and self.P[rb,cb,2]<0 and P[2]<0:
            N = np.cross(self.T[ra,ca]-P,self.T[rb,cb]-P)
            
        if is_null(N):
            return
           
        if ((self.flags & PLANE_LIGHT_EFFECT)>>PLANE_LIGHT_EFFECT_OFFSET) == 0x01:
                E = np.array([0,0,1])
                # R: reflected light
                R = reflection(self.light_T-P,N)
                norm = np.linalg.norm(R)*np.linalg.norm(E-P)
                if norm==0:
                    if np.dot(N,E)>0:
                        red,green,blue = self.color_top
                    else:
                        red,green,blue = self.color_bottom
                else:
                    r = np.dot(R,E-P)/norm
                    r = 1-np.sqrt(0.5*(1-r))
                    r = 0.333 + 0.666*r
                    if np.dot(N,E)>0:
                        red,green,blue = self.color_top
                    else:
                        red,green,blue = self.color_bottom
                    red = int(r*red)
                    green = int(r*green)
                    blue = int(r*blue)
        else:
            if np.dot(N,E)>0:
                red,green,blue = self.color_top
            else:
                red,green,blue = self.color_bottom

        P = PlaneObject([ [self.P[ra,ca,0], self.P[ra,ca,1]],
                          [self.P[rb,cb,0], self.P[rb,cb,1]],
                          [Pp[0], Pp[1]]],
                          self.flags, self.color_line, (red,green,blue),
                          0.333*(self.P[ra,ca,2]+self.P[rb,cb,2]+P[2]))
        planes.append(P)

    def AddPlanes4(self, planes, R):
        for i in range(self.m-1):
            for j in range(self.n-1):
                E = np.array([0,0,1])
                # All four vertices are valid
                if not(self.invalid[i,j] or self.invalid[i,j+1] or self.invalid[i+1,j+1] or self.invalid[i+1,j]):
                    W = 0.25*(self.T[i,j]+self.T[i,j+1]+self.T[i+1,j+1]+self.T[i+1,j])
                    Wp = project_3d(vp, W)
                    self.AddTriangle(i,   j,   i,   j+1, W, Wp, vp, planes)
                    self.AddTriangle(i,   j+1, i+1, j+1, W, Wp, vp, planes)
                    self.AddTriangle(i+1, j+1, i+1, j,   W, Wp, vp, planes)
                    self.AddTriangle(i+1, j,   i,   j,   W, Wp, vp, planes)
                else:
                    if not(self.invalid[i,j] or self.invalid[i,j+1] or self.invalid[i+1,j+1]):
                        W = self.T[i,j+1]
                        Wp = project_3d(vp, W)
                        self.AddTriangle(i+1,   j+1,   i,   j, W, Wp, vp, planes)
                        
                    if not(self.invalid[i,j+1] or self.invalid[i+1,j+1] or self.invalid[i+1,j]):
                        W = self.T[i+1,j+1]
                        Wp = project_3d(vp, W)
                        self.AddTriangle(i+1,   j,   i,   j+1, W, Wp, vp, planes)

                    if not(self.invalid[i+1,j+1] or self.invalid[i+1,j] or self.invalid[i,j]):
                        W = self.T[i+1,j]
                        Wp = project_3d(vp, W)
                        self.AddTriangle(i,   j,   i+1,   j+1, W, Wp, vp, planes)
            
                    if not(self.invalid[i,j] or self.invalid[i,j+1] or self.invalid[i+1,j]):
                        W = self.T[i,j]
                        Wp = project_3d(vp, W)
                        self.AddTriangle(i,   j+1,   i+1,   j, W, Wp, vp, planes)


def ProcessTriangle(A0, A1, A2, vp3d, Rot, flags, color_line, color_top, color_bottom, planes_list):
    T0 = transform_3d(vp3d, Rot, A0)
    T1 = transform_3d(vp3d, Rot, A1)
    T2 = transform_3d(vp3d, Rot, A2)
    
    P0 = project_3d(vp3d,T0)
    P1 = project_3d(vp3d,T1)
    P2 = project_3d(vp3d,T2)

    E = np.array([0,0,1])

    if P0[2]<0 and P1[2]<0 and P2[2]<0:
        N = np.cross(T2-T0, T2-T1)

        if is_null(N):
            return
             
        if ((flags & PLANE_LIGHT_EFFECT)>>PLANE_LIGHT_EFFECT_OFFSET) == 0x01:
                # R: reflected light
                W = 0.3333*(T0+T1+T2)
                light_T = transform_3d(vp3d, Rot, vp3d.light)
                R = reflection(light_T-W,N)
                norm = np.linalg.norm(R)*np.linalg.norm(E-W)
                if norm==0:
                    if np.dot(N,E)>0:
                        red,green,blue = color_top
                    else:
                        red,green,blue = color_bottom
                else:
                    r = np.dot(R,E-W)/norm
                    r = 1-np.sqrt(0.5*(1-r))
                    r = 0.333+0.666*r
                    if np.dot(N,E)>0:
                        red,green,blue = color_top
                    else:
                        red,green,blue = color_bottom
                    red = int(r*red)
                    green = int(r*green)
                    blue = int(r*blue)
        else:
            if np.dot(N,E)>0:
                red,green,blue = color_top
            else:
                red,green,blue = color_bottom

        P = PlaneObject([ [P0[0], P0[1]], [P1[0], P1[1]], [P2[0], P2[1]] ], flags, color_line, (red, green, blue), 0.3333*(P0[2]+P1[2]+P2[2]))
        planes_list.append(P)
  
def ProcessSquare(A0, A1, A2, A3, vp3d, Rot, flags, color_line, color_top, color_bottom, planes_list):
    W = 0.25*(np.array(A0)+np.array(A1)+np.array(A2)+np.array(A3))
    ProcessTriangle(W, A0, A1, vp3d, Rot, flags, color_line, color_top, color_bottom, planes_list)
    ProcessTriangle(W, A1, A2, vp3d, Rot, flags, color_line, color_top, color_bottom, planes_list)
    ProcessTriangle(W, A2, A3, vp3d, Rot, flags, color_line, color_top, color_bottom, planes_list)
    ProcessTriangle(W, A3, A0, vp3d, Rot, flags, color_line, color_top, color_bottom, planes_list)
    
class MathImplicitSurface:
    def __init__(self, size_x, size_y, size_z):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.invalid = np.zeros((size_x, size_y, size_z), dtype='bool')
        self.R =  np.zeros((self.size_x, self.size_y, self.size_z))
        self.Fi = np.zeros((self.size_x-1, self.size_y, self.size_z), dtype='bool')
        self.Fj = np.zeros((self.size_x, self.size_y-1, self.size_z), dtype='bool')
        self.Fk = np.zeros((self.size_x, self.size_y, self.size_z-1), dtype='bool')
        self.Zi = np.zeros((self.size_x-1, self.size_y, self.size_z))
        self.Zj = np.zeros((self.size_x, self.size_y-1, self.size_z))
        self.Zk = np.zeros((self.size_x, self.size_y, self.size_z-1))
        self.Ti = np.zeros((self.size_x-1, self.size_y, self.size_z))
        self.Tj = np.zeros((self.size_x, self.size_y-1, self.size_z))
        self.Tk = np.zeros((self.size_x, self.size_y, self.size_z-1))
                
    def CubeTest(self, V):
        self.size_x = 2
        self.size_y = 2
        self.size_z = 2
        self.R =  np.zeros((self.size_x, self.size_y, self.size_z))
        self.xa = 0
        self.xb = 1
        self.ya = 0
        self.yb = 1
        self.za = 0
        self.zb = 1
        self.dx = 1
        self.dy = 1
        self.dz = 1
        self.R[0,0,0] = V[0]
        self.R[1,0,0] = V[1]
        self.R[0,1,0] = V[2]
        self.R[1,1,0] = V[3]
        self.R[0,0,1] = V[4]
        self.R[1,0,1] = V[5]
        self.R[0,1,1] = V[6]
        self.R[1,1,1] = V[7]

    def LoadMedical(self, filename, size_x, size_y, size_z):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.R =  np.zeros((self.size_x, self.size_y, self.size_z))
        self.xa = -1.0
        self.xb = 1.0
        self.ya = -1.0
        self.yb = 1.0
        self.za = -1.0
        self.zb = 1.0
        self.dx = (self.xb-self.xa)/self.size_x
        self.dy = (self.yb-self.ya)/self.size_y
        self.dz = (self.zb-self.za)/self.size_z
        file = open(filename, "r")
        while True:
            line = file.readline()
            if not line:
                break
            args = line.split()
            if len(args)==4:
#                print(line)
                self.R[int(args[0]),int(args[1]),self.size_z-int(args[2])-1] = float(args[3])
        file.close()
        print("Medical loaded")
        
    def CalcScalarField(self, f, xa, xb, ya, yb, za, zb):
        self.xa = xa
        self.xb = xb
        self.ya = ya
        self.yb = yb
        self.za = za
        self.zb = zb
        self.dx = (xb-xa)/(self.size_x-1)
        self.dy = (yb-ya)/(self.size_y-1)
        self.dz = (zb-za)/(self.size_z-1)
        z = za
        for k in range(self.size_z):
            y = ya
            for j in range(self.size_y):
                x = xa
                for i in range(self.size_x):
                    self.R[i,j,k], self.invalid[i,j,k] = f(x,y,z)
                    x = x + self.dx
                y = y + self.dy
            z = z + self.dz

    def FindZeros(self):
        dx = (self.xb-self.xa)/(self.size_x-1)
        dy = (self.yb-self.ya)/(self.size_y-1)
        dz = (self.zb-self.za)/(self.size_z-1)
        z = self.za
        for k in range(self.size_z):
            y = self.ya
            for j in range(self.size_y):
                x = self.xa
                for i in range(self.size_x):
                    if i<self.size_x-1:
                        self.Fi[i,j,k], self.Zi[i,j,k] = FindZero(x,self.R[i,j,k],x+dx,self.R[i+1,j,k])
                    if j<self.size_y-1:
                        self.Fj[i,j,k], self.Zj[i,j,k] = FindZero(y,self.R[i,j,k],y+dy,self.R[i,j+1,k])
                    if k<self.size_z-1:
                        self.Fk[i,j,k], self.Zk[i,j,k] = FindZero(z,self.R[i,j,k],z+dz,self.R[i,j,k+1])
                    x = x + dx
                y = y + dy
            z = z + dz

    def RotateX(self, codes):
        temp_c = codes[0]
        codes[0]=codes[4]
        codes[4]=codes[6]
        codes[6]=codes[2]
        codes[2]=temp_c
        
        temp_c = codes[1]
        codes[1]=codes[5]
        codes[5]=codes[7]
        codes[7]=codes[3]
        codes[3]=temp_c

    def RotateY(self, codes):
        temp_c = codes[0]
        codes[0]=codes[1]
        codes[1]=codes[5]
        codes[5]=codes[4]
        codes[4]=temp_c
        
        temp_c = codes[2]
        codes[2]=codes[3]
        codes[3]=codes[7]
        codes[7]=codes[6]
        codes[6]=temp_c
        
    def RotateZ(self, codes):
        temp_c = codes[0]
        codes[0]=codes[2]
        codes[2]=codes[3]
        codes[3]=codes[1]
        codes[1]=temp_c
        
        temp_c = codes[4]
        codes[4]=codes[6]
        codes[6]=codes[7]
        codes[7]=codes[5]
        codes[5]=temp_c

    def P(self, code, i, j, k):
        if code==0:
            return np.array([self.xa+self.dx*i,self.ya+self.dy*j,self.za+self.dz*k])
        elif code==1:
            return np.array([self.xa+self.dx*(i+1),self.ya+self.dy*j,self.za+self.dz*k])
        elif code==2:
            return np.array([self.xa+self.dx*i,self.ya+self.dy*(j+1),self.za+self.dz*k])
        elif code==3:
            return np.array([self.xa+self.dx*(i+1),self.ya+self.dy*(j+1),self.za+self.dz*k])
        elif code==4:
            return np.array([self.xa+self.dx*i,self.ya+self.dy*j,self.za+self.dz*(k+1)])
        elif code==5:
            return np.array([self.xa+self.dx*(i+1),self.ya+self.dy*j,self.za+self.dz*(k+1)])
        elif code==6:
            return np.array([self.xa+self.dx*i,self.ya+self.dy*(j+1),self.za+self.dz*(k+1)])
        elif code==7:
            return np.array([self.xa+self.dx*(i+1),self.ya+self.dy*(j+1),self.za+self.dz*(k+1)])

    def Q(self, code, i, j, k):
        if code==0:
            return self.R[i,j,k]-self.iso_value
        elif code==1:
            return self.R[i+1,j,k]-self.iso_value
        elif code==2:
            return self.R[i,j+1,k]-self.iso_value
        elif code==3:
            return self.R[i+1,j+1,k]-self.iso_value
        elif code==4:
            return self.R[i,j,k+1]-self.iso_value
        elif code==5:
            return self.R[i+1,j,k+1]-self.iso_value
        elif code==6:
            return self.R[i,j+1,k+1]-self.iso_value
        elif code==7:
            return self.R[i+1,j+1,k+1]-self.iso_value
        
    def ProcessIntersections(self,vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list, iso_value):
        self.iso_value=iso_value
        dx = (self.xb-self.xa)/(self.size_x-1)
        dy = (self.yb-self.ya)/(self.size_y-1)
        dz = (self.zb-self.za)/(self.size_z-1)
        z = self.za
        for k in range(self.size_z-1):
            y = self.ya
            for j in range(self.size_y-1):
                x = self.xa
                for i in range(self.size_x//3,self.size_x-1,1):
                    FLAG_DEBUG=False
                    v = [0, 1, 2, 3, 4, 5, 6, 7 ]
                    s = [self.Q(0,i,j,k)>=0, self.Q(1,i,j,k)>=0, self.Q(2,i,j,k)>=0, self.Q(3,i,j,k)>=0,
                         self.Q(4,i,j,k)>=0, self.Q(5,i,j,k)>=0, self.Q(6,i,j,k)>=0, self.Q(7,i,j,k)>=0 ]

                    if FLAG_DEBUG:
                        print("------------------------------------------")
                        print(s)
                        print(t)
                        surface.fill((255, 255, 255))
                    count_rot = 0
                    while count_rot<64:
                        if s[0]!=s[1]:
                            valid, c01 = FindZeroVec(self.P(v[0],i,j,k), self.Q(v[0],i,j,k), self.P(v[1],i,j,k), self.Q(v[1],i,j,k))
                            
                        if s[0]!=s[2]:
                            valid, c02 = FindZeroVec(self.P(v[0],i,j,k), self.Q(v[0],i,j,k), self.P(v[2],i,j,k), self.Q(v[2],i,j,k))
                            
                        if s[0]!=s[4]:
                            valid, c04 = FindZeroVec(self.P(v[0],i,j,k), self.Q(v[0],i,j,k), self.P(v[4],i,j,k), self.Q(v[4],i,j,k))

                        if s[1]!=s[3]:
                            valid, c13 = FindZeroVec(self.P(v[1],i,j,k), self.Q(v[1],i,j,k), self.P(v[3],i,j,k), self.Q(v[3],i,j,k))

                        if s[1]!=s[5]:
                            valid, c15 = FindZeroVec(self.P(v[1],i,j,k), self.Q(v[1],i,j,k), self.P(v[5],i,j,k), self.Q(v[5],i,j,k))

                        if s[3]!=s[2]:
                            valid, c23 = FindZeroVec(self.P(v[3],i,j,k), self.Q(v[3],i,j,k), self.P(v[2],i,j,k), self.Q(v[2],i,j,k))

                        if s[3]!=s[7]:
                            valid, c37 = FindZeroVec(self.P(v[3],i,j,k), self.Q(v[3],i,j,k), self.P(v[7],i,j,k), self.Q(v[7],i,j,k))

                        if s[2]!=s[6]:
                            valid, c26 = FindZeroVec(self.P(v[2],i,j,k), self.Q(v[2],i,j,k), self.P(v[6],i,j,k), self.Q(v[6],i,j,k))

                        if s[4]!=s[5]:
                            valid, c45 = FindZeroVec(self.P(v[4],i,j,k), self.Q(v[4],i,j,k), self.P(v[5],i,j,k), self.Q(v[5],i,j,k))

                        if s[4]!=s[6]:
                            valid, c46 = FindZeroVec(self.P(v[4],i,j,k), self.Q(v[4],i,j,k), self.P(v[6],i,j,k), self.Q(v[6],i,j,k))

                        if s[5]!=s[7]:
                            valid, c57 = FindZeroVec(self.P(v[5],i,j,k), self.Q(v[5],i,j,k), self.P(v[7],i,j,k), self.Q(v[7],i,j,k))

                        if s[7]!=s[6]:
                            valid, c67 = FindZeroVec(self.P(v[7],i,j,k), self.Q(v[7],i,j,k), self.P(v[6],i,j,k), self.Q(v[6],i,j,k))

                        # Caso 0
                        if s[0] and s[1] and s[2] and s[3] and s[4] and s[5] and s[6] and s[7]:
                            break
                    
                        if not(s[0]) and not(s[1]) and not(s[2]) and not(s[3]) and not(s[4]) and not(s[5]) and not(s[6]) and not(s[7]) :
                            break

                        # Caso 1
                        if s[0] and not(s[1]) and not(s[2]) and not(s[3]) and not(s[4]) and not(s[5]) and not(s[6]) and not(s[7]):
                            ProcessTriangle( c01, c04, c02, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso01")
                            break

                        if not(s[0]) and s[1] and s[2] and s[3] and s[4] and s[5] and s[6] and s[7]:
                            ProcessTriangle( c01, c02, c04, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso01c")
                            break

                        # Caso 2
                        if s[0] and s[4] and not(s[1]) and not(s[2]) and not(s[3]) and not(s[5]) and not(s[6]) and not(s[7]):
                            ProcessTriangle( c01, c46, c02, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c01, c45, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso02")
                            break

                        if not(s[0]) and not(s[4]) and s[1] and s[2] and s[3] and s[5] and s[6] and s[7]:
                            ProcessTriangle( c01, c02, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c01, c46, c45, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso02c")
                            break

                        # Caso 3
                        if s[0] and s[6] and not(s[1]) and not(s[2]) and not(s[3]) and not(s[4]) and not(s[5]) and not(s[7]):
                            ProcessTriangle( c01, c04, c02, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c26, c46, c67, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso03")
                            break

                        if not(s[0]) and not(s[6]) and s[1] and s[2] and s[3] and s[4] and s[5] and s[7]:
                            ProcessTriangle( c01, c02, c04, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c26, c67, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso03c")
                            break

                        # Caso 4
                        if s[0] and s[7] and not(s[1]) and not(s[2]) and not(s[3]) and not(s[4]) and not(s[5]) and not(s[6]):
                            ProcessTriangle( c01, c04, c02, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c37, c67, c57, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso04")
                            break

                        if not(s[0]) and not(s[7]) and s[1] and s[2] and s[3] and s[4] and s[5] and s[6]:
                            ProcessTriangle( c01, c02, c04, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c37, c57, c67, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso04c")
                            break

                        # Caso 5
                        if s[0] and s[2] and s[3] and not(s[1]) and not(s[4]) and not(s[5]) and not(s[6]) and not(s[7]):
                            ProcessTriangle( c13, c01, c37, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c37, c01, c04, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c37, c04, c26, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso05")
                            break
                        
                        if not(s[0]) and not(s[2]) and not(s[3]) and s[1] and s[4] and s[5] and s[6] and s[7]:
                            ProcessTriangle( c13, c37, c01, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c37, c04, c01, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c37, c26, c04, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso05c")
                            break
                        
                        # Caso 6
                        if s[1] and s[3] and s[6] and not(s[0]) and not(s[2]) and not(s[4]) and not(s[5]) and not(s[7]):
                            ProcessTriangle( c23, c37, c01, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c01, c37, c15, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c67, c26, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso06")
                            break

                        if not(s[1]) and not(s[3]) and not(s[6]) and s[0] and s[2] and s[4] and s[5] and s[7]:
                            ProcessTriangle( c23, c01, c37, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c01, c15, c37, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c67, c46, c26, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso06c")
                            break

                         # Caso 7
                        if s[0] and s[5] and s[6] and not(s[1]) and not(s[2]) and not(s[3]) and not(s[4]) and not(s[7]):
                            ProcessTriangle( c01, c04, c02, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c57, c45, c15, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c67, c26, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso07")
                            break

                        if not(s[0]) and not(s[5]) and not(s[6]) and s[1] and s[2] and s[3] and s[4] and s[7]:
                            ProcessTriangle( c01, c02, c04, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c57, c15, c45, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c67, c46, c26, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso07c")
                            break

                        # Caso 8
                        if s[0] and s[1] and s[2] and s[3] and not(s[4]) and not(s[5]) and not(s[6]) and not(s[7]):
                            ProcessTriangle( c37, c15, c26, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c04, c26, c15, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso08")
                            break

                        if not(s[0]) and not(s[1]) and not(s[2]) and not(s[3]) and s[4] and s[5] and s[6] and s[7]:
                            ProcessTriangle( c37, c26, c15, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c04, c15, c26, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso08c")
                            break

                        # Caso9
                        if s[0] and s[1] and s[2] and s[4] and not(s[3]) and not(s[5]) and not(s[6]) and not(s[7]):
                            ProcessTriangle( c13, c15, c45, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c13, c45, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c13, c46, c23, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c46, c26, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            # print("caso09")
                            break

                        if not(s[0]) and not(s[1]) and not(s[2]) and not(s[4]) and s[3] and s[5] and s[6] and s[7]:
                            ProcessTriangle( c13, c45, c15, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c13, c46, c45, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c13, c23, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c26, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            #print("caso09c")
                            break

                        # Caso 10
                        if s[1] and s[2] and s[5] and s[6] and not(s[0]) and not(s[3]) and not(s[4]) and not(s[7]):
                            ProcessTriangle( c13, c57, c45, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c13, c45, c01, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c46, c67, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c02, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso10")
                            break

                        if not(s[1]) and not(s[2]) and not(s[5]) and not(s[6]) and s[0] and s[3] and s[4] and s[7]:
                            ProcessTriangle( c13, c45, c57, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c13, c01, c45, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c67, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c46, c02, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso10c")
                            break

                        # Caso 11
                        if s[0] and s[1] and s[2] and s[6] and not(s[3]) and not(s[4]) and not(s[5]) and not(s[7]):
                            ProcessTriangle( c13, c15, c04, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c13, c04, c23, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c04, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c46, c67, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso11")
                            break

                        if not(s[0]) and not(s[1]) and not(s[2]) and not(s[6]) and s[3] and s[4] and s[5] and s[7]:
                            ProcessTriangle( c13, c04, c15, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c13, c23, c04, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c46, c04, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c67, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso11c")
                            break

                        # Caso12
                        if s[0] and s[1] and s[3] and s[6] and not(s[2]) and not(s[4]) and not(s[5]) and not(s[7]):
                            ProcessTriangle( c37, c15, c04, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c37, c04, c02, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c37, c02, c23, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c67, c26, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso12")
                            break

                        if not(s[0]) and not(s[1]) and not(s[3]) and not(s[6]) and s[2] and s[4] and s[5] and s[7]:
                            ProcessTriangle( c37, c04, c15, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c37, c02, c04, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c37, c23, c02, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c67, c46, c26, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso12c")
                            break

                        # Caso13
                        if s[0] and s[3] and s[5] and s[6] and not(s[1]) and not(s[2]) and not(s[4]) and not(s[7]):
                            ProcessTriangle( c01, c04, c02, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c13, c23, c37, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c67, c26, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c15, c57, c45, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso13")
                            break

                        if not(s[0]) and not(s[3]) and not(s[5]) and not(s[6]) and s[1] and s[2] and s[4] and s[7]:
                            ProcessTriangle( c01, c04, c02, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c13, c23, c37, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c67, c26, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c15, c57, c45, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso13c")
                            break

                        # Caso14
                        if s[0] and s[1] and s[2] and s[6] and not(s[3]) and not(s[4]) and not(s[5]) and not(s[7]):
                            ProcessTriangle( c23, c13, c15, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c15, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c46, c67, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso14")
                            break
                        
                        if not(s[0]) and not(s[1]) and not(s[2]) and not(s[6]) and s[3] and s[4] and s[5] and s[7]:
                            ProcessTriangle( c23, c15, c13, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c46, c15, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
                            ProcessTriangle( c23, c67, c46, vp3d, Rot,  flags, color_line, color_top, color_bottom, planes_list)
#                            print("caso14c")
                            break

                        count_rot = count_rot+1
    
                        self.RotateZ(v)
                        self.RotateZ(s)
                        #print("Rotated z")
                        if count_rot%4==0:
                            self.RotateY(v)
                            self.RotateY(s)
                        #    print("Rotated y")

                        if count_rot%16==0:
                            self.RotateX(v)
                            self.RotateX(s)
                         #   print("Rotated x")

                #    if count_rot!=0:
                #        print(count_rot)

                    if FLAG_DEBUG:
                        DrawPainterPlanes(planes_list)
                        draw_box_3d([x,y,z],[x+dx,y+dy,z+dz], vp3d, Rot, 3, "black")
                        pygame.display.update()
                        d = input()
                     
                    x = x + dx
                y = y + dy
            z = z + dz
##          if k%10 ==0:
##        print("ProcessIntersections()","{:.1f}".format(100*k/self.size_z),"%")
##        print("ProcessIntersections() completed")
#         PrintMatriz3D("Zi=",Zi)
##        print("Zj")
##        PrintMatriz3D(Zj)
##        print("Zk")
##        PrintMatriz3D(Zk)
##        print(Zi)

def determinant_3x3(matrix):
    return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
            matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
            matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))

def solve_3x3_system(A, B):
    det_A = determinant_3x3(A)
    if det_A == 0:
        return [0,0,0], False
    
    # Create matrices A_x, A_y, A_z by replacing respective columns with B
    A_x = [[B[0], A[0][1], A[0][2]],
           [B[1], A[1][1], A[1][2]],
           [B[2], A[2][1], A[2][2]]]
    
    A_y = [[A[0][0], B[0], A[0][2]],
           [A[1][0], B[1], A[1][2]],
           [A[2][0], B[2], A[2][2]]]
    
    A_z = [[A[0][0], A[0][1], B[0]],
           [A[1][0], A[1][1], B[1]],
           [A[2][0], A[2][1], B[2]]]

    # Calculate the determinants of A_x, A_y, A_z
    det_A_x = determinant_3x3(A_x)
    det_A_y = determinant_3x3(A_y)
    det_A_z = determinant_3x3(A_z)

    # Calculate the values of x, y, z using Cramer's rule
    x = det_A_x / det_A
    y = det_A_y / det_A
    z = det_A_z / det_A

    return [x, y, z], True

# Example usage
# A = [[2, -1, 3],
#      [4, 2, -1],
#      [-6, -3, 2]]
#B = [5, -2, 1]
#solution = solve_3x3_system(A, B)
#print("Solution:", solution)

def GetBoxTriangle(A,B,C):
    min = np.copy(A)
    max = np.copy(A)
    if B[0]<min[0]:   min[0]=B[0]
    if B[1]<min[1]:   min[1]=B[1]
    if B[2]<min[2]:   min[2]=B[2]

    if B[0]>max[0]:   max[0]=B[0]
    if B[1]>max[1]:   max[1]=B[1]
    if B[2]>max[2]:   max[2]=B[2]

    if C[0]<min[0]:   min[0]=C[0]
    if C[1]<min[1]:   min[1]=C[1]
    if C[2]<min[2]:   min[2]=C[2]

    if C[0]>max[0]:   max[0]=C[0]
    if C[1]>max[1]:   max[1]=C[1]
    if C[2]>max[2]:   max[2]=C[2]
    return min, max

def GetBoxLine(A,B):
    min = np.copy(A)
    max = np.copy(A)
    if B[0]<min[0]:   min[0]=B[0]
    if B[1]<min[1]:   min[1]=B[1]
    if B[2]<min[2]:   min[2]=B[2]

    if B[0]>max[0]:   max[0]=B[0]
    if B[1]>max[1]:   max[1]=B[1]
    if B[2]>max[2]:   max[2]=B[2]
    return min, max

def InterTriangleAndLine(A0, A1, A2, B0, B1):
    min_T, max_T = GetBoxTriangle(A0, A1, A2)
    min_L, max_L = GetBoxLine(B0, B1)
    # Detect whether the segments' boxes do not overlap.  If they do, there is no need to get the intersection
    if (max_L[0]<min_T[0] and max_L[1]<min_T[1] and max_L[2]<min_T[2]) or (min_L[0]>max_T[0] and min_L[1]>max_T[1] and min_L[2]>max_T[2]):  
         return np.array([0,0,0]), False

    M = np.zeros((3,3))
    M[0,0] = A1[0]-A0[0];   M[0,1] = A2[0]-A0[0];   M[0,2] = B0[0]-B1[0] 
    M[1,0] = A1[1]-A0[1];   M[1,1] = A2[1]-A0[1];   M[1,2] = B0[1]-B1[1] 
    M[2,0] = A1[2]-A0[2];   M[2,1] = A2[2]-A0[2];   M[2,2] = B0[2]-B1[2]

    C = np.zeros(3)
    C[0] = B0[0] - A0[0]
    C[1] = B0[1] - A0[1]
    C[2] = B0[2] - A0[2]
    X, flag = solve_3x3_system(M, C)
    if not(flag):
        return X, flag

    flag_is_in_triangle = (0.0<=X[0] and X[0]<=1.0) and (0.0<=X[1] and X[1]<=1.0) and (X[0] + X[1]<=1.0) and (0.0<=X[2] and X[2]<=1.0)
    return B0 + X[2]*(B1 - B0), flag_is_in_triangle

A0 = np.array([0,0,1])
A1 = np.array([1,0,1])
A2 = np.array([1,1,1])
B0 = np.array([0.5,0.51,-2])
B1 = np.array([0.5,0.51,2])

# print("InterTriangleAndLine = ",InterTriangleAndLine(A0, A1, A2, B0, B1))

def InterTriangles(A0, A1, A2, B0, B1, B2, segments_list):
    P_initial = [0,0,0]
    
    flag_first = True
    
    P, flag  = InterTriangleAndLine(A0, A1, A2, B0, B1)

    if flag:
        P_initial = P
        flag_first = False
    
    P, flag = InterTriangleAndLine(A0, A1, A2, B0, B2)
    if flag:
        if flag_first:
            P_initial = P
            flag_first = False
        else:
            segments_list.append([P_initial, P])
            return
        
    P, flag =  InterTriangleAndLine(A0, A1, A2, B1, B2)
    if flag:
        if flag_first:
            P_initial = P
            flag_first = False
        else:
            segments_list.append([P_initial, P])
            return
        
    P, flag =  InterTriangleAndLine(B0, B1, B2, A0, A1)
    if flag:
        if flag_first:
            P_initial = P
            flag_first = False
        else:
            segments_list.append([P_initial, P])
            return

    P, flag =  InterTriangleAndLine(B0, B1, B2, A0, A2)
    if flag:
        if flag_first:
            P_initial = P
            flag_first = False
        else:
            segments_list.append([P_initial, P])
            return

    P, flag =  InterTriangleAndLine(B0, B1, B2, A1, A2)
    if flag:
        if flag_first:
            P_initial = P
            flag_first = False
        else:
            segments_list.append([P_initial, P])
            return
    return

def CalculateIntersections(Sa, Sb, segments_list):
    for ia in range(Sa.m-1):
        for ja in range(Sa.n-1):
            for ib in range(Sb.m-1):
                for jb in range(Sb.n-1):
                    count_int = 0
                    
                    # Intersection of TB1 in TA1
                    P_initial = [0,0,0]

                    P_L0_in_A1, flag_L0_in_A1  = InterTriangleAndLine(Sa.R[ia,ja],  Sa.R[ia,ja+1], Sa.R[ia+1,ja+1], Sb.R[ib,jb], Sb.R[ib+1,jb+1])
                    P_M0_in_B1, flag_M0_in_B1  = InterTriangleAndLine(Sb.R[ib,jb],  Sb.R[ib,jb+1], Sb.R[ib+1,jb+1], Sa.R[ia,ja], Sa.R[ia+1,ja+1])
                   
                    if flag_L0_in_A1:
                        P_initial = P_L0_in_A1
                        count_int = count_int + 1

                    if flag_M0_in_B1:
                        if count_int==0:
                            P_initial = P_M0_in_B1
                        if count_int==1:
                            segments_list.append([P_initial, P_M0_in_B1])
                        count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sa.R[ia,ja],  Sa.R[ia,ja+1], Sa.R[ia+1,ja+1], Sb.R[ib,jb],        Sb.R[ib,jb+1])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sa.R[ia,ja],  Sa.R[ia,ja+1], Sa.R[ia+1,ja+1], Sb.R[ib,jb+1],      Sb.R[ib+1,jb+1])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sb.R[ib,jb],  Sb.R[ib,jb+1], Sb.R[ib+1,jb+1], Sa.R[ia,ja],      Sa.R[ia,ja+1])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1
          
                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sb.R[ib,jb],  Sb.R[ib,jb+1], Sb.R[ib+1,jb+1], Sa.R[ia,ja+1],      Sa.R[ia+1,ja+1])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    ################################################
                    count_int = 0

                    if flag_L0_in_A1:
                        P_initial = P_L0_in_A1
                        count_int = 1
                        
                    P_M0_in_B2, flag_M0_in_B2  = InterTriangleAndLine(Sb.R[ib,jb],  Sb.R[ib+1,jb+1], Sb.R[ib+1,jb], Sa.R[ia,ja], Sa.R[ia+1,ja+1])
                    if flag_M0_in_B2:
                        if count_int==0:
                            P_initial = P_M0_in_B2
                        if count_int==1:
                            segments_list.append([P_initial, P_M0_in_B2])
                        count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sa.R[ia,ja],  Sa.R[ia,ja+1], Sa.R[ia+1,ja+1], Sb.R[ib+1,jb], Sb.R[ib+1,jb+1])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sa.R[ia,ja],  Sa.R[ia,ja+1], Sa.R[ia+1,ja+1], Sb.R[ib+1,jb], Sb.R[ib,jb])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1


                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sb.R[ib,jb],  Sb.R[ib+1,jb+1], Sb.R[ib+1,jb], Sa.R[ia,ja], Sa.R[ia,ja+1])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1
                            
                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sb.R[ib,jb],  Sb.R[ib+1,jb+1], Sb.R[ib+1,jb], Sa.R[ia,ja+1], Sa.R[ia+1,ja+1])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    ################################################
                    count_int = 0

                    if flag_M0_in_B1:
                        P_initial = P_M0_in_B1
                        count_int = 1

                    P_L0_in_A2, flag_L0_in_A2  = InterTriangleAndLine(Sa.R[ia,ja],  Sa.R[ia+1,ja+1], Sa.R[ia+1,ja], Sb.R[ib,jb], Sb.R[ib+1,jb+1])
                    if flag_L0_in_A2:
                        if count_int==0:
                            P_initial = P_L0_in_A2
                        if count_int==1:
                            segments_list.append([P_initial, P_L0_in_A2])
                        count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sa.R[ia,ja],  Sa.R[ia+1,ja+1], Sa.R[ia+1,ja], Sb.R[ib,jb], Sb.R[ib,jb+1])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sa.R[ia,ja],  Sa.R[ia+1,ja+1], Sa.R[ia+1,ja], Sb.R[ib,jb+1], Sb.R[ib+1,jb+1])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sb.R[ib,jb],  Sb.R[ib,jb+1], Sb.R[ib+1,jb+1], Sa.R[ia+1,ja+1], Sa.R[ia+1,ja])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sb.R[ib,jb],  Sb.R[ib,jb+1], Sb.R[ib+1,jb+1], Sa.R[ia+1,ja], Sa.R[ia,ja])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    ################################################
                    count_int = 0

                    if flag_L0_in_A2:
                        P_initial = P_L0_in_A2
                        count_int = 1

                    if flag_M0_in_B2:
                        if count_int==0:
                            P_initial = P_M0_in_B2
                        if count_int==1:
                            segments_list.append([P_initial, P_M0_in_B2])
                        count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sa.R[ia,ja],  Sa.R[ia+1,ja+1], Sa.R[ia+1,ja], Sb.R[ib+1,jb+1], Sb.R[ib+1,jb])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sa.R[ia,ja],  Sa.R[ia+1,ja+1], Sa.R[ia+1,ja], Sb.R[ib+1,jb], Sb.R[ib,jb])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sb.R[ib,jb],  Sb.R[ib+1,jb+1], Sb.R[ib+1,jb], Sa.R[ia+1,ja+1], Sa.R[ia+1,ja])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1

                    if count_int<2:
                        P, flag  = InterTriangleAndLine(Sb.R[ib,jb],  Sb.R[ib+1,jb+1], Sb.R[ib+1,jb], Sa.R[ia+1,ja], Sa.R[ia,ja])
                        if flag:
                            if count_int==0:
                                P_initial = P
                            if count_int==1:
                                segments_list.append([P_initial, P])
                            count_int = count_int + 1
        print("CalculateIntersections = ", '{0:.1f}'.format(100*ia/(Sa.m-1)),"%")
    print("CalculateIntersections = ", '{0:.1f}'.format(100*ia/(Sa.m-1)),"%")


def CalculateIntersections2(Sa, Sb, segments_list):
    for ia in range(Sa.m-1):
        for ja in range(Sa.n-1):
            for ib in range(Sb.m-1):
                for jb in range(Sb.n-1):
                    InterTriangles(Sa.R[ia,ja], Sa.R[ia,ja+1], Sa.R[ia+1,ja+1],  Sb.R[ib,jb], Sb.R[ib,jb+1], Sb.R[ib+1,jb+1],   segments_list)
                    InterTriangles(Sa.R[ia,ja], Sa.R[ia,ja+1], Sa.R[ia+1,ja+1],  Sb.R[ib,jb], Sb.R[ib+1,jb], Sb.R[ib+1,jb+1],   segments_list)
                    InterTriangles(Sa.R[ia,ja], Sa.R[ia+1,ja], Sa.R[ia+1,ja+1],  Sb.R[ib,jb], Sb.R[ib,jb+1], Sb.R[ib+1,jb+1],   segments_list)
                    InterTriangles(Sa.R[ia,ja], Sa.R[ia+1,ja], Sa.R[ia+1,ja+1],  Sb.R[ib,jb], Sb.R[ib+1,jb], Sb.R[ib+1,jb+1],   segments_list)
                    
        print("CalculateIntersections2 = ", '{0:.1f}'.format(100*ia/(Sa.m-1)),"%")
    print("CalculateIntersections2 = ", '{0:.1f}'.format(100),"%")


def CalculateIntersections4(Sa, Sb, segments_list):
    for ia in range(Sa.m-1):
        for ja in range(Sa.n-1):
            Wa = 0.25*(Sa.R[ia,ja]+Sa.R[ia,ja+1]+Sa.R[ia+1,ja+1]+Sa.R[ia+1,ja])
            for ib in range(Sb.m-1):
                for jb in range(Sb.n-1):
                    Wb = 0.25*(Sb.R[ib,jb]+Sb.R[ib,jb+1]+Sb.R[ib+1,jb+1]+Sb.R[ib+1,jb])
                    InterTriangles(Wa, Sa.R[ia,ja], Sa.R[ia,ja+1], Wb, Sb.R[ib,jb],     Sb.R[ib,jb+1],   segments_list)
                    InterTriangles(Wa, Sa.R[ia,ja], Sa.R[ia,ja+1], Wb, Sb.R[ib,jb+1],   Sb.R[ib+1,jb+1], segments_list)
                    InterTriangles(Wa, Sa.R[ia,ja], Sa.R[ia,ja+1], Wb, Sb.R[ib+1,jb+1], Sb.R[ib+1,jb],   segments_list)
                    InterTriangles(Wa, Sa.R[ia,ja], Sa.R[ia,ja+1], Wb, Sb.R[ib+1,jb],   Sb.R[ib,jb],     segments_list)
                  
                    InterTriangles(Wa, Sa.R[ia,ja+1], Sa.R[ia+1,ja+1], Wb, Sb.R[ib,jb],     Sb.R[ib,jb+1],   segments_list)
                    InterTriangles(Wa, Sa.R[ia,ja+1], Sa.R[ia+1,ja+1], Wb, Sb.R[ib,jb+1],   Sb.R[ib+1,jb+1], segments_list)
                    InterTriangles(Wa, Sa.R[ia,ja+1], Sa.R[ia+1,ja+1], Wb, Sb.R[ib+1,jb+1], Sb.R[ib+1,jb],   segments_list)
                    InterTriangles(Wa, Sa.R[ia,ja+1], Sa.R[ia+1,ja+1], Wb, Sb.R[ib+1,jb],   Sb.R[ib,jb],     segments_list)

                    InterTriangles(Wa, Sa.R[ia+1,ja+1], Sa.R[ia+1,ja], Wb, Sb.R[ib,jb],     Sb.R[ib,jb+1],   segments_list)
                    InterTriangles(Wa, Sa.R[ia+1,ja+1], Sa.R[ia+1,ja], Wb, Sb.R[ib,jb+1],   Sb.R[ib+1,jb+1], segments_list)
                    InterTriangles(Wa, Sa.R[ia+1,ja+1], Sa.R[ia+1,ja], Wb, Sb.R[ib+1,jb+1], Sb.R[ib+1,jb],   segments_list)
                    InterTriangles(Wa, Sa.R[ia+1,ja+1], Sa.R[ia+1,ja], Wb, Sb.R[ib+1,jb],   Sb.R[ib,jb],     segments_list)

                    InterTriangles(Wa, Sa.R[ia+1,ja], Sa.R[ia,ja], Wb, Sb.R[ib,jb],     Sb.R[ib,jb+1],   segments_list)
                    InterTriangles(Wa, Sa.R[ia+1,ja], Sa.R[ia,ja], Wb, Sb.R[ib,jb+1],   Sb.R[ib+1,jb+1], segments_list)
                    InterTriangles(Wa, Sa.R[ia+1,ja], Sa.R[ia,ja], Wb, Sb.R[ib+1,jb+1], Sb.R[ib+1,jb],   segments_list)
                    InterTriangles(Wa, Sa.R[ia+1,ja], Sa.R[ia,ja], Wb, Sb.R[ib+1,jb],   Sb.R[ib,jb],     segments_list)
        print("CalculateIntersections4 = ", '{0:.1f}'.format(100*ia/(Sa.m-1)),"%")
    print("CalculateIntersections4 = ", '{0:.1f}'.format(100),"%")

# if not(Sa.invalid[i,j] or Sa.invalid[i,j+1] or Sa.invalid[i+1,j+1] or Sa.invalid[i+1,j]):

# Curves
def spiral(t):
    return [ 0.1*t*np.cos(t), 0.1*t*np.sin(t), 0.25*t ], False 

def line_k(t):
    return [ 0, 0, t ], False 

# Surfaces
def plane1(t,u):
    return [ t, u, 1 ], False 

def plane2(t,u):
    return [ t, u, 0.2 ], False 

def plane_a(t,u):
    return [ t, u, -1.5 ], False 

def plane_b(t,u):
    return [ t, u, -0.33*t+1.5 ], False 

def plane_c(t,u):
    return [ 0.5-t, 0.5+t, u ], False 

##def plane_d(t,u):
##    return [-1.5+ t*0.707-u, -1.5+t*0.707+u, -1.5+t], False

def otho_versors(A):
    A = np.array(A)

    A_norm = np.linalg.norm(A)
    if A_norm == 0:
        if A[0] != 0 or A[1] != 0:
            B = np.array([-A[1], A[0], 0])
        else:
            B = np.array([1, 0, 0])  # A[2] must be non-zero, so choose a simple perpendicular vector
        B = B / np.linalg.norm(B)
        C = np.cross(A, B)
        C = C / np.linalg.norm(C)
        return B, C
    else:
         return np.array([0,0,0]), np.array([0,0,0])


def plane_d(t,u):
    return [1.5- t*0.707-u, 1.5-t*0.707+u, t], False

def plane_e(t,u):
    return [1, t, u], False

def plane_f(t,u):
    return [t, u, 0.5], False

def plane_g(t,u):
    return [t, u, 1], False

def plane_h(t,u):
    return [t, u, 1.5], False

def plane_i(t,u):
    return [t, u, -0.5], False

def plane_j(t,u):
    return [t, u, -1], False

def plane_k(t,u):
    return [t, u, -1.5], False


def paraboloid(t,u):
    return [ t, u, (t**2+u**2)/4 ], False 

def sphere(t,u):
    return [ np.cos(t)*np.sin(u), np.sin(t)*np.sin(u), np.cos(u) ], False 

def semi_sphere(t,u):
    r = 1-t**2 - u**2
    if r<0:
        return [0,0,0], True
    return [ t, u, np.sqrt(r) ], False 

def mobius(t,u):
    r = 1.25+0.25*np.cos(0.5*t)*u
    return [ np.cos(t)*r, np.sin(t)*r,np.sin(0.5*t)*u ] , False

def cone(t,u):
    return [ u*np.cos(t), u*np.sin(t), u ], False 

def cylinder(t,u):
    return [ np.cos(t), np.sin(t), t ], False

def klein_bottle(u,v):
    # u : [0, pi]
    # v : [0, 2pi]
    cu = np.cos(u)
    su = np.sin(u)
    cv = np.cos(v)
    sv = np.sin(v)
    
    x = -(2.0/15.0)*cu*(3*cv-30*su+90*cu**4*su-60*cu**6*su+5*cu*cv*su)
    y = -(2.0/15.0)*su*(3*cv-3*cu**2*cv-48*cu**4*cv+48*cu**6*cv-60*su+5*cu*cv*su-5*cu**3*cv*su-80*cu**5*cv*su+80*cu**7*cv*su)
    z =  (2.0/15.0)*(3+5*cu*su)*sv  
    return [ x, y, z ], False

def ortho_versors(A):
    # Convert the input vector A to a numpy array and normalize it
    A_norm = np.linalg.norm(A)
    if A_norm != 0:
        A = A / A_norm
        if A[0] != 0 or A[1] != 0:
            B = np.array([-A[1], A[0], 0])
        else:
            B = np.array([1, 0, 0])
        B = B / np.linalg.norm(B)
        C = np.cross(A, B)
        C = C / np.linalg.norm(C)
        return B, C
    else:
        return np.array([0,0,0]), np.array([0,0,0])
    
def bell(t):
    return np.array([3*np.sin(t), 3*np.sin(t)**2*np.cos(t),0])

def klein_bell(t,tetha):
    delta = 0.001
    P = bell(t)
    A = (bell(t+delta)-P)/delta
    B, C = ortho_versors(A)
    r = 0.5 - (1.0/30.0)*(2*t-np.pi)*np.sqrt(2*t*(2*np.pi-2*t))
    return P + r*B*np.cos(tetha) + r*C*np.sin(tetha), False

def torus(t,u):
    a = 0.5
    b = 2
    return [ (a*np.cos(t) + b)*np.cos(u), (a*np.cos(t) + b)*np.sin(u),a*np.sin(t) ] , False

def circle_e3(t):
    A = np.array([1 ,1, 0])
    A = A/np.linalg.norm(A)
    B = np.array([1 ,0, 0])
    B = B/np.linalg.norm(B)
#    return A
    return A*np.cos(t)+B*np.sin(t), False 

def triang_1(t,u):
    P0 = np.array([0,0,0])
    P1 = np.array([3,0,0])
    P2 = np.array([3,3,0])
    return P0 + t*(P1-P0)+(1-t)*u*(P2-P0), False

def sphere_imp(x,y,z):
    return x**2+y**2+z**2-1, False

def toris_imp(x,y,z):
    R = 1
    a = 0.2
    r = 0.01
    t = (x**2 + y**2 + z**2 + R**2 -a**2)**2
    F1 =t -4*R**2*(x**2+y**2)
    F2 =t -4*R**2*(x**2+z**2)
    F3 =t -4*R**2*(y**2+z**2)
    return F1*F2*F3-r, False
   
def cylinder_imp(x,y,z):
    return x**2+y**2-1, False

def family_esc(x,y):
  #  return x**3 -3*x+y**3-3*y , False
    # return np.sin(x**2+y**2), False
  return 5*x*np.exp(-x**2-y**2), False

def family(x,y):
    #return [x, y, x**3 -3*x+y**3-3*y ], False
    #return [x,y,0], False
    #return [x,y,0.5*np.sin(x**2+y**2)], False
    return [x,y,5*x*np.exp(-x**2-y**2)], False

def AddPlanesAxis(P0,P1, axis_radius, arrow_radius, arrrow_length, vp, R, planes, color_line, color_top, color_bottom, flags):
    Axis = MathSurface(7,10, color_line, color_top, color_bottom, flags)

    A = P1-P0
    if is_null(A):
        return

    if A[0]!=0:
        norm = np.sqrt(((A[1]+A[2])/A[0])**2+2)
        B = np.array([(-A[1]-A[2])/A[0], 1,1])/norm
        C = np.cross(B, A/np.linalg.norm(A))
    elif A[1]!=0:
        norm = np.sqrt(((A[0]+A[2])/A[1])**2+2)
        B = np.array([1, (-A[0]-A[2])/A[1], 1])/norm
        C = np.cross(B, A/np.linalg.norm(A))
    else:
        norm = np.sqrt(((A[1]+A[0])/A[2])**2+2)
        B = np.array([1,1,(-A[1]-A[0])/A[2]])/norm
        C = np.cross(B, A/np.linalg.norm(A))

    B = axis_radius*B
    C = axis_radius*C

    du = 1.0/(Axis.m-1)
    dt = 2*np.pi/(Axis.n-1)
    u = 0
    for i in range(Axis.m):
        t = 0
        for j in range(Axis.n):
            Axis.R[i,j] = P0+u*(P1-P0) + B*np.cos(t) + C*np.sin(t)
            t = t + dt
        u = u + du

    Axis.Transform(vp,R)
    Axis.Project(vp,R)
    Axis.AddPlanes(planes)
    
    B = B*arrow_radius/axis_radius
    C = C*arrow_radius/axis_radius
    Arrow = MathSurface(10,20, color_line, color_top, color_bottom, flags)

    du = 1.0/(Arrow.m-1)
    dt = 2*np.pi/(Arrow.n-1)
    u = 0

    D = arrrow_length*(P1-P0)/np.linalg.norm(P1-P0)
    for i in range(Arrow.m):
        t = 0
        for j in range(Arrow.n):
            Arrow.R[i,j] = P1+D-u*D + u*B*np.cos(t) + u*C*np.sin(t)
            t = t + dt
        u = u + du
    
    Arrow.Transform(vp,R)
    Arrow.Project(vp,R)
    Arrow.AddPlanes(planes)

def PrintMatriz3D(legend, M):
    size_x, size_y, size_z = np.shape(M)
    print(legend) 
    for k in range(size_z):
        print("k=",k,"-------------------")
        for j in range(size_y):
            for i in range(size_x):
                print("{:6.2f}".format(M[i,j,k]), "  ", end='')
            print()
     
# Orthogonal component of B in A
def OrtoComponent(A,B):
    norm_sq = np.dot(A,A)
    if norm_sq == 0:
        return np.array([0,0,0])
    return B - np.dot(B,A)*A/norm_sq

def AxialComponent(A,B):
    norm_sq = np.dot(A,A)
    if norm_sq == 0:
        return np.array([0,0,0])
    return np.dot(B,A)*A/norm_sq

C1 = MathCurve(1000, (0,0,255), PLANE_LIGHT_EFFECT+3)
C1.Calc(line_k, -5,5)

def DrawPainterPlanes(surface, my_list):
    #planes_list_sorted = sorted(my_list, key=lambda x: x.distance)
            
    my_list.sort(key=lambda x: x.distance)
    k =0
    k_max = len(my_list)
    for P in my_list:
        if len(P.geo)==2:
            pygame.draw.line(surface, P.color, P.geo[0],P.geo[1], P.flags & PLANE_BORDER_WIDTH_MASK)
    
        if len(P.geo)>2:
            if ((P.flags & PLANE_FILL)>>PLANE_FILL_OFFSET) == 0x01:
                   pygame.draw.polygon(surface, P.color, P.geo, width=0)
            
            if (P.flags & PLANE_BORDER_WIDTH_MASK)>0:
                    pygame.draw.polygon(surface, P.color_border, P.geo, width=P.flags & PLANE_BORDER_WIDTH_MASK)
##        if k % (k_max//100) == 0:
##            print("DrawPainterPlanes()","{:.1f}".format(100*k/k_max))
##        k = k + 1
                    
def DrawSegments(surface, vp, R_cam, segments_list, color, width):
    for segment in segments_list:
        P0 = plot_3d(vp, R_cam, segment[0])
        P1 = plot_3d(vp, R_cam, segment[1])
        pygame.draw.line(surface, color, [P0[0], P0[1]], [P1[0], P1[1]], width)

screen = pygame.display.set_mode((screen_width, screen_height))

if False:
    screen.fill((255, 255, 255))
    # Creacion del Viewport 2D
    vp2d = VP2d()
    vp2d.xp_min=50
    vp2d.xp_max=screen_width-50
    vp2d.yp_min=200
    vp2d.yp_max=screen_height-200
    vp2d.xr_min=-3.0
    vp2d.xr_max=3.0
    vp2d.yr_min=-2.0
    vp2d.yr_max=2.0

    draw_rectangle_2d(screen, vp2d,[-2.5,-1.66],[2.5,1.66],2,(0,0,0))
    draw_vector_2d(screen, vp2d, [-2.5,0], [2.5,0], 15, 30*D2R, 2, (0,0,0))
    draw_vector_2d(screen, vp2d, [0,-1.66], [0,1.66], 15, 30*D2R, 2, (0,0,0))

    ImplicitCurves = MathImplicitCurves(200,200)
    ImplicitCurves.CalcScalarField(family_esc, -2.5, 2.5, -1.66, 1.66)
   # ImplicitCurves.FindZeros(vp2d)
   
    ImplicitCurves.WalkingSquare(screen, vp2d, -3.5, 3, (0,0,200))
    ImplicitCurves.WalkingSquare(screen, vp2d, -3, 3, (0,0,200))
    ImplicitCurves.WalkingSquare(screen, vp2d, -2.5, 3, (0,0,200))
    ImplicitCurves.WalkingSquare(screen, vp2d, -2, 3, (0,0,200))
    ImplicitCurves.WalkingSquare(screen, vp2d, -1.5, 3, (0,0,200))
    ImplicitCurves.WalkingSquare(screen, vp2d, -1, 3, (0,0,200))
    ImplicitCurves.WalkingSquare(screen, vp2d, -0.5, 3, (0,0,200))
    ImplicitCurves.WalkingSquare(screen, vp2d, 0, 3, (0,0,0))
    ImplicitCurves.WalkingSquare(screen, vp2d, 3.5, 3, (200,0,0))
    ImplicitCurves.WalkingSquare(screen, vp2d, 3, 3, (200,0,0))
    ImplicitCurves.WalkingSquare(screen, vp2d, 2.5, 3, (200,0,0))
    ImplicitCurves.WalkingSquare(screen, vp2d, 2, 3, (200,0,0))
    ImplicitCurves.WalkingSquare(screen, vp2d, 1.5, 3, (200,0,0))
    ImplicitCurves.WalkingSquare(screen, vp2d, 1, 3, (200,0,0))
    ImplicitCurves.WalkingSquare(screen, vp2d, 0.5, 3, (200,0,0))
    pygame.display.update()
    time.sleep(120)

while False:
    #2.949606435870417 4.50294947014537 4.468042885105484 2.6878070480712677 0.3490658503988659 0.6981317007977318
    a1=D2R*random.randint(0, 360)
    a2=D2R*random.randint(0, 360)
    a3=D2R*random.randint(0, 360)
    a4=D2R*random.randint(0, 360)
    a5=D2R*random.randint(0, 360)
    a6=D2R*random.randint(0, 360)
    print(a1,a2,a3,a4,a5,a6)
               
    screen.fill((255, 255, 255))
    # Creacion del Viewport 2D
    screen.fill((255, 255, 255))
    planes_list = []

    R1 = create_rot_xy(a1)
    R2 = create_rot_xz(a2)
    R3 = create_rot_xw(a3)
    R4 = create_rot_yz(a4)
    R5 = create_rot_yw(a5)
    R6 = create_rot_zw(a6)

    R4d = R1 @ R2 @ R3 @ R4 @ R5 @ R6

    Pa = np.array([0,0,0,0])
    Pb = np.array([1,0,0,0])
    Pc = np.array([1,1,0,0])
    Pd = np.array([0,1,0,0])

    Pe = np.array([0,0,1,0])
    Pf = np.array([1,0,1,0])
    Pg = np.array([1,1,1,0])
    Ph = np.array([0,1,1,0])

    Pi = np.array([0,0,0,1])
    Pj = np.array([1,0,0,1])
    Pk = np.array([1,1,0,1])
    Pl = np.array([0,1,0,1])

    Pm = np.array([0,0,1,1])
    Pn = np.array([1,0,1,1])
    Po = np.array([1,1,1,1])
    Pp = np.array([0,1,1,1])

    xc = screen_width/2
    yc = screen_height/2
    scale = 250
    A = plot_4d_2d(xc, yc, scale, R4d, Pa)
    B = plot_4d_2d(xc, yc, scale, R4d, Pb)
    C = plot_4d_2d(xc, yc, scale, R4d, Pc)
    D = plot_4d_2d(xc, yc, scale, R4d, Pd)
    E = plot_4d_2d(xc, yc, scale, R4d, Pe)
    F = plot_4d_2d(xc, yc, scale, R4d, Pf)
    G = plot_4d_2d(xc, yc, scale, R4d, Pg)
    H = plot_4d_2d(xc, yc, scale, R4d, Ph)
    I = plot_4d_2d(xc, yc, scale, R4d, Pi)
    J = plot_4d_2d(xc, yc, scale, R4d, Pj)
    K = plot_4d_2d(xc, yc, scale, R4d, Pk)
    L = plot_4d_2d(xc, yc, scale, R4d, Pl)
    M = plot_4d_2d(xc, yc, scale, R4d, Pm)
    N = plot_4d_2d(xc, yc, scale, R4d, Pn)
    O = plot_4d_2d(xc, yc, scale, R4d, Po)
    P = plot_4d_2d(xc, yc, scale, R4d, Pp)

    pygame.draw.line(screen, (0,0,0), A, B, 3)
    pygame.draw.line(screen, (0,0,0), B, C, 3)
    pygame.draw.line(screen, (0,0,0), C, D, 3)
    pygame.draw.line(screen, (0,0,0), D, A, 3)

    pygame.draw.line(screen, (0,0,0), E, F, 3)
    pygame.draw.line(screen, (0,0,0), F, G, 3)
    pygame.draw.line(screen, (0,0,0), G, H, 3)
    pygame.draw.line(screen, (0,0,0), H, E, 3)

    pygame.draw.line(screen, (0,0,0), A, E, 3)
    pygame.draw.line(screen, (0,0,0), B, F, 3)
    pygame.draw.line(screen, (0,0,0), C, G, 3)
    pygame.draw.line(screen, (0,0,0), D, H, 3)

    pygame.draw.line(screen, (0,0,255), I, J, 3)
    pygame.draw.line(screen, (0,0,255), J, K, 3)
    pygame.draw.line(screen, (0,0,255), K, L, 3)
    pygame.draw.line(screen, (0,0,255), L, I, 3)

    pygame.draw.line(screen, (0,0,255), M, N, 3)
    pygame.draw.line(screen, (0,0,255), N, O, 3)
    pygame.draw.line(screen, (0,0,255), O, P, 3)
    pygame.draw.line(screen, (0,0,255), P, M, 2)

    pygame.draw.line(screen, (0,0,255), I, M, 3)
    pygame.draw.line(screen, (0,0,255), J, N, 3)
    pygame.draw.line(screen, (0,0,255), K, O, 3)
    pygame.draw.line(screen, (0,0,255), L, P, 3)

    pygame.draw.line(screen, (0,128,0), A, I, 3)
    pygame.draw.line(screen, (0,128,0), B, J, 3)
    pygame.draw.line(screen, (0,128,0), C, K, 3)
    pygame.draw.line(screen, (0,128,0), D, L, 3)

    pygame.draw.line(screen, (0,128,0), E, M, 3)
    pygame.draw.line(screen, (0,128,0), H, P, 3)
    pygame.draw.line(screen, (0,128,0), F, N, 3)
    pygame.draw.line(screen, (0,128,0), G, O, 3)

    
    pygame.display.flip()

    input()
    for ev in pygame.event.get():
        if ev.type == QUIT:
            pygame.quit()
            sys.exit()



##########################################################################################
#yaw, pitch, roll = ( -1.5820 , 0.2158 , 0.5595 ) #skull
yaw, pitch, roll = ( -0.3510 , 1.0199 , -1.9886 ) # chest
#yaw, pitch, roll = ( -1.2670 , 0.6848 , -2.9426) #chest with view to the abdomen
R_cam = create_tait_bryan_ZYX(yaw, pitch, roll)
##########################################################################################
vp3d = VP3d()
vp3d.o_x   = screen_width/2
vp3d.o_y   = screen_height/2
vp3d.scale = 650
#vp3d.scale = 240
vp3d.cam = np.array([0, 0, 10])
vp3d.cam_d = 0
vp3d.cam_h = 20
vp3d.iso = False
vp3d.light = np.array([30 ,0, 5])
ImplicitSurface = MathImplicitSurface(1,1,1)
print("LoadMedical started")
ImplicitSurface.LoadMedical("chest_512_512_173.txt",512,512,173)
#ImplicitSurface.LoadMedical("chest_128_128_173.txt",128,128,173)
#ImplicitSurface.LoadMedical("chest_64_64_173.txt",64,64,173)
#ImplicitSurface.LoadMedical("skull_512_512_109.txt",512,512,109)
#ImplicitSurface.LoadMedical("skull_256_256_108.txt",256,256,108)
#ImplicitSurface.LoadMedical("skull_128_128_108.txt",128,128,108)
#ImplicitSurface.LoadMedical("skull_64_64_54.txt",64,64,54)
#ImplicitSurface.LoadMedical("skull_32_32_54.txt",32,32,54)
print("LoadMedical completed")
#ImplicitSurface.CubeTest([-1, -1, -1, 1, 1, 1, -1, 1])
#ImplicitSurface.CalcScalarField(toris_imp, -1.25, 1.25, -1.25, 1.25, -1.25, 1.25)

# ffmpeg -framerate 30 -i IMG%04d.bmp -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4
def process_image(value):
    flag_debug = True
    print("Started processing",value)
    surface = pygame.Surface((screen_width,screen_height))
    planes_list = []
    surface.fill((255, 255, 255))
    if flag_debug:
        print("Started ProcessIntersections()")
    ImplicitSurface.ProcessIntersections(vp3d, R_cam, 0 | PLANE_FILL | PLANE_LIGHT_EFFECT, (0,0,0), (246,241,196),  (246,241,196), planes_list, value)
    if flag_debug:
        print("Completed ProcessIntersections()")
        print("Started DrawPainterPlanes()")
    DrawPainterPlanes(surface, planes_list)
    if flag_debug:
        print("Started DrawPainterPlanes()")
    draw_text(surface,10, 10,str(value),"black")
    filename=r"d:\temp8\IMG"+"{:04d}".format(value)+".bmp"
    pygame.image.save(surface,filename)
            

flag = True
if __name__ == "__main__" and flag:
    max_processes  = 24
    processes = []
    print("Program started")

    for k in range(500, 501,1):
        process = multiprocessing.Process(target=process_image, args=(k,))
        processes.append(process)
        process.start()
        
        if len(processes) >= max_processes:
            for process in processes:
                process.join()
            processes = []

    for process in processes:
       process.join()
    print("Program completed")


#while (True):
#    time.sleep(1)


# Ciclo infinito

flag_draw = True
flag_changed = True
pygame.key.set_repeat(0, 100)

#yaw, pitch, roll =  (0.3062, 0.7976, -1.9987)
#yaw, pitch, roll = (-45*D2R,-90*D2R,-45*D2R)
#yaw, pitch, roll = (-2.7720, -0.6975, 1.0166)
#yaw, pitch, roll = ( -2.9618 , -0.4739 , 1.2111 ) #conos
#yaw, pitch, roll = ( 3.0310 , 0.6150 , 1.3706 ) # cono tangente
#yaw, pitch, roll = ( 3.1030 , 0.2389 , 1.3838 ) # parabola
#yaw, pitch, roll = ( -0.6444 , 0.3785 , -2.6771 ) #klein
#yaw, pitch, roll = ( -2.9518 , -0.7557 , 1.2863 )
#R_cam = create_tait_bryan_ZYX(yaw, pitch, roll)

print("Infinite loop")
while not(flag):
    # Presentar los resultados para los sistemas relativo y fijo.
    if flag_changed:
        yaw, pitch, roll = get_tait_bryan_ZYX(R_cam)
        print("yaw, pitch, roll = (",'{0:.4f}'.format(yaw),",",'{0:.4f}'.format(pitch),",",'{0:.4f}'.format(roll),")")
        R_cam = create_tait_bryan_ZYX(yaw, pitch, roll)
        flag_draw=True

    if flag_draw:
        screen.fill((255, 255, 255))
        print("cleared")
        planes_list = []

        # Enable/disable fancy axis
        if True:
            axis_size=0.02
            axis_length = 1.0
            axis_color = (40,40,40)
            AddPlanesAxis(np.array([0,0,0]),np.array([axis_length,0,0]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)
            AddPlanesAxis(np.array([0,0,0]),np.array([0,axis_length,0]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)
            AddPlanesAxis(np.array([0,0,0]),np.array([0,0,axis_length]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)

            Px = plot_3d(vp3d, R_cam, np.array([axis_length,0,0]))
            draw_text(screen,Px[0]+10, Px[1]-55,"X","black")
            Py = plot_3d(vp3d, R_cam, np.array([0,axis_length,0]))
            draw_text(screen,Py[0]+10, Py[1]-55,"Y","black")
            Pz = plot_3d(vp3d, R_cam, np.array([0,0,axis_length]))
            draw_text(screen,Pz[0]+10, Pz[1]-55,"Z","black")
            DrawPainterPlanes(screen, planes_list)
                
        # Enable/disable basic axis
        if False:
            axis_length = 1.0
            axis_color = (0,0,255)
            draw_vector_3d(screen, [0,0,0],[axis_length,0,0], vp3d, R_cam, 10, 30*D2R, 5, axis_color)
            draw_vector_3d(screen, [0,0,0],[0,axis_length,0], vp3d, R_cam, 10, 30*D2R, 5, axis_color)
            draw_vector_3d(screen, [0,0,0],[0,0,axis_length], vp3d, R_cam, 10, 30*D2R, 5, axis_color)
            
    flag_changed=False
 
    for ev in pygame.event.get():
        if ev.type == QUIT:
            pygame.quit()
            sys.exit()
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_UP:    
                R_cam =  np.matmul(Rp, R_cam)
                flag_changed=True

            if ev.key == pygame.K_DOWN:
                R_cam =  np.matmul(Rm, R_cam)
                flag_changed=True

            if ev.key == pygame.K_LEFT:
                R_cam =  np.matmul(Yp, R_cam)
                flag_changed=True

            if ev.key == pygame.K_RIGHT:
                R_cam =  np.matmul(Ym, R_cam)
                flag_changed=True

            if ev.key == pygame.K_PAGEUP:
                R_cam =  np.matmul(Pp, R_cam)
                flag_changed=True

            if ev.key == pygame.K_PAGEDOWN:
                R_cam =  np.matmul(Pm, R_cam)
                flag_changed=True

            if ev.key == pygame.K_z:
                vp3d.cam[2]=vp3d.cam[2]-1
                print("z = ",vp3d.cam[2])
                flag_changed=True

            if ev.key == pygame.K_x:
                vp3d.cam[2]=vp3d.cam[2]+1
                print("z=",vp3d.cam[2])
                flag_changed=True

            if ev.key == pygame.K_F1:
                shift = pygame.key.get_mods()& pygame.KMOD_SHIFT
                if shift:
                    vp3d.light =  vp3d.light + [1,0,0]
                else:
                    vp3d.light =  vp3d.light + [-1,0,0]
                print("Light=",vp.light)
                flag_changed=True

            if ev.key == pygame.K_F2:
                shift = pygame.key.get_mods()& pygame.KMOD_SHIFT
                if shift:
                    vp.light =  vp.light + [0,1,0]
                else:
                    vp.light =  vp.light + [0,-1,0]
                print("Light=",vp.light)
                flag_changed=True

            if ev.key == pygame.K_F3:
                shift = pygame.key.get_mods()& pygame.KMOD_SHIFT
                if shift:
                    vp.light =  vp.light + [0,0,1]
                else:
                    vp.light =  vp.light + [0,0,-1]
                print("Light=",vp.light)
                flag_changed=True

            if ev.key == pygame.K_a:
                screen.fill((255, 255, 255))
                planes_list = []

                axis_size=0.02
                axis_length = 3
                axis_color = (0,0,0)
                AddPlanesAxis(np.array([0,0,0]),np.array([axis_length,0,0]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)
                AddPlanesAxis(np.array([0,0,0]),np.array([0,axis_length,0]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)
                AddPlanesAxis(np.array([0,0,0]),np.array([0,0,axis_length]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)

                S1 = MathSurface(40,40,(0,0,0), (255,255,0), (255,255,255), 1 | PLANE_FILL | PLANE_LIGHT_EFFECT)
                S1.Calc(family,-2,2,-2,2)
                S1.Transform(vp3d,R_cam)
                S1.Project(vp3d,R_cam)
                S1.AddPlanes(planes_list)
               
                S4 = MathSurface(2,2, (0,0,0), (0,0,255), (0,0,0), 2 | 0*PLANE_FILL | 1*PLANE_LIGHT_EFFECT)
                S4.Calc(plane_f,-2,2,-2,2)
                S4.Transform(vp3d,R_cam)
                S4.Project(vp3d,R_cam)
                S4.AddPlanes(planes_list)

                S5 = MathSurface(2,2, (0,0,0), (0,0,255), (0,0,0), 2 | 0*PLANE_FILL | 1*PLANE_LIGHT_EFFECT)
                S5.Calc(plane_g,-2,2,-2,2)
                S5.Transform(vp3d,R_cam)
                S5.Project(vp3d,R_cam)
                S5.AddPlanes(planes_list)

                S6 = MathSurface(2,2, (0,0,0), (0,0,255), (0,0,0), 2 | 0*PLANE_FILL | 1*PLANE_LIGHT_EFFECT)
                S6.Calc(plane_h,-2,2,-2,2)
                S6.Transform(vp3d,R_cam)
                S6.Project(vp3d,R_cam)
                S6.AddPlanes(planes_list)

                S7 = MathSurface(2,2, (0,0,0), (0,0,255), (0,0,0), 2 | 0*PLANE_FILL | 1*PLANE_LIGHT_EFFECT)
                S7.Calc(plane_i,-2,2,-2,2)
                S7.Transform(vp3d,R_cam)
                S7.Project(vp3d,R_cam)
                S7.AddPlanes(planes_list)

                S8 = MathSurface(2,2, (0,0,0), (0,0,255), (0,0,0), 2 | 0*PLANE_FILL | 1*PLANE_LIGHT_EFFECT)
                S8.Calc(plane_j,-2,2,-2,2)
                S8.Transform(vp3d,R_cam)
                S8.Project(vp3d,R_cam)
                S8.AddPlanes(planes_list)

                S9 = MathSurface(2,2, (0,0,0), (0,0,255), (0,0,0), 2 | 0*PLANE_FILL | 1*PLANE_LIGHT_EFFECT)
                S9.Calc(plane_k,-2,2,-2,2)
                S9.Transform(vp3d,R_cam)
                S9.Project(vp3d,R_cam)
                S9.AddPlanes(planes_list)

                DrawPainterPlanes(screen,planes_list)

   
                segments_list = []
                CalculateIntersections4(S1, S4, segments_list)
                DrawSegments(screen,vp3d,R_cam,segments_list, (0,0,255), 4)

                segments_list = []
                CalculateIntersections4(S1, S5, segments_list)
                DrawSegments(screen,vp3d,R_cam,segments_list, (0,0,255), 4)

                segments_list = []
                CalculateIntersections4(S1, S6, segments_list)
                DrawSegments(screen,vp3d,R_cam,segments_list, (0,0,255), 4)

                segments_list = []
                CalculateIntersections4(S1, S7, segments_list)
                DrawSegments(screen,vp3d,R_cam,segments_list, (255,255,255), 4)
                            
                segments_list = []
                CalculateIntersections4(S1, S8, segments_list)
                DrawSegments(screen,vp3d,R_cam,segments_list, (255,255,255), 4)

                segments_list = []
                CalculateIntersections4(S1, S9, segments_list)
                DrawSegments(screen,vp3d,R_cam,segments_list, (255,255,255), 4)

                Px = plot_3d(vp3d, R_cam, np.array([axis_length,0,0]))
                draw_text(screen,Px[0]-10, Px[1]+15,"X","black")
                Py = plot_3d(vp3d, R_cam, np.array([0,axis_length,0]))
                draw_text(screen,Py[0]+10, Py[1]+15,"Y","black")
                Pz = plot_3d(vp3d, R_cam, np.array([0,0,axis_length]))
                draw_text(screen,Pz[0]+10, Pz[1]+15,"Z","black")

                flag_draw=True


            if ev.key == pygame.K_b:
                screen.fill((255, 255, 255))
                planes_list = []

                axis_size=0.02
                axis_length = 4.0
                axis_color = (40,40,40)
                AddPlanesAxis(np.array([0,0,0]),np.array([axis_length,0,0]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)
                AddPlanesAxis(np.array([0,0,0]),np.array([0,axis_length,0]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)
                AddPlanesAxis(np.array([0,0,0]),np.array([0,0,axis_length]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)

              
                print("len(planes_list) before=",len(planes_list))
                q = 5
                S1 = MathSurface(q*10,q*10,(0,0,0), (255,0,0), (0,0,255), 1 | PLANE_FILL | PLANE_LIGHT_EFFECT)
                S1.Calc(family,-2.5,2.5,-2.5,2.5)
                #S1.Calc(mobius,0,2*np.pi,-0.25,0.25)

                S1.Transform(vp3d,R_cam)
                S1.Project(vp3d,R_cam)
                S1.AddPlanes(planes_list)
                print("len(planes_list) after=",len(planes_list))
               
                S2 = MathSurface(40,40, (0,0,0), (255,255,0), (255,255,0), 0 | 1*PLANE_FILL | 1*PLANE_LIGHT_EFFECT)
                S2.Calc(plane1,-2.5,2.5,-2.5,2.5)
                S2.Transform(vp3d,R_cam)
                S2.Project(vp3d,R_cam)
                S2.AddPlanes(planes_list)

           
                #S2.Calc(klein_bottle,0,np.pi,0,2*np.pi)
                #S2.Calc(torus,0,2*np.pi,0,2*np.pi)
                #S2.Calc(paraboloid,-2.5, 2.5, -2.5, 2.5)

                segments_list = []
                CalculateIntersections2(S1, S2, segments_list)
                DrawPainterPlanes(screen,planes_list)
                            
                DrawSegments(screen,vp3d,R_cam,segments_list, (0,0,0), 3)

                Px = plot_3d(vp3d, R_cam, np.array([axis_length,0,0]))
                draw_text(screen,Px[0]+10, Px[1]-55,"X","black")
                Py = plot_3d(vp3d, R_cam, np.array([0,axis_length,0]))
                draw_text(screen,Py[0]+10, Py[1]-55,"Y","black")
                Pz = plot_3d(vp3d, R_cam, np.array([0,0,axis_length]))
                draw_text(screen,Pz[0]+10, Pz[1]-55,"Z","black")

                flag_draw=True

            if ev.key == pygame.K_c:
                screen.fill((255, 255, 255))
                planes_list = []

                axis_size=0.02
                axis_length = 3.0
                axis_color = (0,0,0)
                AddPlanesAxis(np.array([0,0,0]),np.array([axis_length,0,0]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)
                AddPlanesAxis(np.array([0,0,0]),np.array([0,axis_length,0]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)
                AddPlanesAxis(np.array([0,0,0]),np.array([0,0,axis_length]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)

                q=5
                S1 = MathSurface(10*q,10*q,(0,0,0), (0,0,255), (175,175,175), 0 | PLANE_FILL | PLANE_LIGHT_EFFECT)
                S1.Calc(cone,0,6.2832,-2.5,2.5)
                S1.Transform(vp3d,R_cam)
                S1.Project(vp3d,R_cam)
                S1.AddPlanes(planes_list)

               
                S3 = MathSurface(10*(q-1),10*(q-1), (0,0,0), (255,255,0), (255,255,0), 0 | 1*PLANE_FILL | 1*PLANE_LIGHT_EFFECT)
                S3.Calc(plane_a,-2.5,2.5,-3,3)
                S3.Transform(vp3d,R_cam)
                S3.Project(vp3d,R_cam)
                S3.AddPlanes(planes_list)

                segments_list = []
                CalculateIntersections4(S1, S3, segments_list)
                DrawPainterPlanes(screen,planes_list)
                            
                DrawSegments(screen,vp3d,R_cam,segments_list, (255,255,255), 3)

                Px = plot_3d(vp3d, R_cam, np.array([axis_length,0,0]))
                draw_text(screen,Px[0]+10, Px[1]-55,"X","black")
                Py = plot_3d(vp3d, R_cam, np.array([0,axis_length,0]))
                draw_text(screen,Py[0]+10, Py[1]-55,"Y","black")
                Pz = plot_3d(vp3d, R_cam, np.array([0,0,axis_length]))
                draw_text(screen,Pz[0]+10, Pz[1]-55,"Z","black")

                Pa, flag = plane_a(-2.5, 3)
                Pb, flag = plane_a(2.5,   3)
                Pc, flag = plane_a(2.5,  -3)
                Pd, flag = plane_a(-2.5,-3)

                draw_line_3d(screen, vp3d, R_cam, Pa, Pb, 2, (0,0,0))               
                draw_line_3d(screen, vp3d, R_cam, Pb, Pc, 2, (0,0,0))               
                draw_line_3d(screen, vp3d, R_cam, Pc, Pd, 2, (0,0,0))               
                draw_line_3d(screen, vp3d, R_cam, Pd, Pa, 2, (0,0,0))               
                
                flag_draw=True

            if ev.key == pygame.K_d:
                screen.fill((255, 255, 255))
                planes_list = []

                axis_size=0.02
                axis_length = 3.5
                axis_color = (0,0,0)
                AddPlanesAxis(np.array([0,0,0]),np.array([axis_length,0,0]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)
                AddPlanesAxis(np.array([0,0,0]),np.array([0,axis_length,0]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)
                AddPlanesAxis(np.array([0,0,0]),np.array([0,0,axis_length]), axis_size, 2*axis_size, 12*axis_size, vp3d, R_cam, planes_list, axis_color, axis_color, axis_color, 1 | PLANE_FILL | 0*PLANE_LIGHT_EFFECT)

                S1 = MathSurface(90,180,(0,0,0), (175,175,175), (0,0,255), 0 | PLANE_FILL | PLANE_LIGHT_EFFECT)
                #S1 = MathSurface(15,100,(0,0,0), (175,175,175), (0,0,255), 0 | PLANE_FILL | PLANE_LIGHT_EFFECT)
                S1.Calc(klein_bell,0,np.pi,0,2*np.pi)
                S1.Transform(vp3d,R_cam)
                S1.Project(vp3d,R_cam)
                S1.AddPlanes(planes_list)

               
                S3 = MathSurface(20,20, (0,0,0), (255,255,0), (255,255,0), 0 | 1*PLANE_FILL | 1*PLANE_LIGHT_EFFECT)
                S3.Calc(plane_e,-2,2,-1.5,1.5)
                S3.Transform(vp3d,R_cam)
                S3.Project(vp3d,R_cam)
                S3.AddPlanes(planes_list)

                segments_list = []
                CalculateIntersections4(S1, S3, segments_list)
                DrawPainterPlanes(screen,planes_list)
                            
                DrawSegments(screen,vp3d,R_cam,segments_list, (255,255,255), 3)

                Px = plot_3d(vp3d, R_cam, np.array([axis_length,0,0]))
                draw_text(screen,Px[0]+10, Px[1]-55,"X","black")
                Py = plot_3d(vp3d, R_cam, np.array([0,axis_length,0]))
                draw_text(screen,Py[0]+10, Py[1]-55,"Y","black")
                Pz = plot_3d(vp3d, R_cam, np.array([0,0,axis_length]))
                draw_text(screen,Pz[0]+10, Pz[1]-55,"Z","black")

                flag_draw=True

            if ev.key == pygame.K_h:
                flag_debug=True
                isovalue=1000
                print("Started processing")
                surface = pygame.Surface((screen_width,screen_height))
                planes_list = []
                surface.fill((255, 255, 255))
                if flag_debug:
                    print("Started ProcessIntersections()")
                ImplicitSurface.ProcessIntersections(vp3d, R_cam, 0 | PLANE_FILL | PLANE_LIGHT_EFFECT, (0,0,0), (246,241,196),  (246,241,196), planes_list, isovalue)
                if flag_debug:
                    print("Completed ProcessIntersections()")
                    print("Started DrawPainterPlanes()")
                DrawPainterPlanes(surface, planes_list)
                if flag_debug:
                    print("Started DrawPainterPlanes()")
                draw_text(surface,10, 10,str(isovalue),"black")
                #filename=r"d:\temp4\IMG"+"{:04d}".format(isovalue)+".bmp"
                screen.blit(surface, (0,0))
                #pygame.image.save(surface,filename)
                flag_draw=True

            if ev.key == pygame.K_u:
                pygame.display.flip()

    if flag_draw:
        pygame.display.flip()
        flag_draw=False
        flag_changed=False
   

        #ProcessTriangle(vp, R_cam, [0,0,0], [0.3,0,0], [0.3,0.3,0],  1 | PLANE_FILL | PLANE_LIGHT_EFFECT, (0,0,0), (0,0,255), (255,255,0), planes_list)


        # P0 = np.array([5,0,0])
        # P1 = np.array([0,5,0])
        # P2 = np.array([6,7,0])
        # u = 0
        # for i in range(100):
        #   t = 0
        #   for j in range(100):
        #       dot = P0 + t*(P1-P0)+u*(P2-P0)
        #       if (t+u)<=1.0:
        #           plot_dot(vp, R_cam, dot, 1, (255,0,0))
        #       else:
        #           plot_dot(vp, R_cam, dot, 1, (0,255,0))
        #       t = t + 1.0/101
        #   u = u + 1.0/101
        #
        # u = 0
        # Axial =  AxialComponent(P1-P0,P2-P0)
        # for i in range(100):
        #   t = 0
        #   for j in range(100):
        #       dot = np.array([t,u,0])
        #       if np.dot(P2-P0-Axial,dot-P0-Axial)>=0:
        #           plot_dot(vp, R_cam, dot, 1, (255,0,0))
        #       else:
        #           plot_dot(vp, R_cam, dot, 1, (0,0,255))
        #       t = t + 10.0/101
        #   u = u + 10.0/101

        # S1.Transform(vp,R_cam)        # S1.Project(vp,R_cam)
        # S1.AddPlanes(planes_list)
            
        # S2.Transform(vp,R_cam)
        # S2.Project(vp,R_cam)
        # S2.AddPlanes(planes_list)        
      
        # C1.Transform(vp,R_cam)
        # C1.Project(vp,R_cam)
        # C1.AddSegments(vp,planes_list)        
