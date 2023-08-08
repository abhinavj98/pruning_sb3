# Better samples/python2/opt_flow.py

## reference
# - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

## class diagram
# http://www.yuml.me/diagram/scruffy/class/draw
# [IOpticalFlow]^[DenseOpticalFlow],[IOpticalFlow]^[LucasKanadeOpticalFlow], [DenseOpticalFlow]^[DenseOpticalFlowByHSV],[DenseOpticalFlow]^[DenseOpticalFlowByLines],[DenseOpticalFlow]^[DenseOpticalFlowByWarp]

import numpy as np
import cv2
import torchvision
import torch
    from torchvision.models.optical_flow import raft_small,Raft_Small_Weights

import torchvision.transforms.functional as F
# Better samples/python2/opt_flow.py

## reference
# - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html

## class diagram
# http://www.yuml.me/diagram/scruffy/class/draw
# [IOpticalFlow]^[DenseOpticalFlow],[IOpticalFlow]^[LucasKanadeOpticalFlow], [DenseOpticalFlow]^[DenseOpticalFlowByHSV],[DenseOpticalFlow]^[DenseOpticalFlowByLines],[DenseOpticalFlow]^[DenseOpticalFlowByWarp]

import numpy as np
import cv2

class IOpticalFlow:
    '''Interface of OpticalFlow classes'''
    def set1stFrame(self, frame):
        '''Set the starting frame'''
        self.prev = frame

    def apply(self, frame):
        '''Apply and return result display image (expected to be new object)'''
        result = frame.copy()
        self.prev = frame
        return result

class DenseOpticalFlow(IOpticalFlow):
    '''Abstract class for DenseOpticalFlow expressions'''
    def set1stFrame(self, frame):
        self.prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.hsv = np.zeros_like(frame)
        self.hsv[..., 1] = 255

    def apply(self, frame):
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(self.prev, next, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        result = self.makeResult(next, flow)
        self.prev = next
        return result

    def makeResult(self, grayFrame, flow):
        '''Replace this for each expression'''
        return frame.copy()

class DenseOpticalFlowByHSV(DenseOpticalFlow):
    def makeResult(self, grayFrame, flow):
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        self.hsv[...,0] = ang*180/np.pi/2
        self. hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

class DenseOpticalFlowByLines(DenseOpticalFlow):
    def __init__(self):
        self.step = 16 # configure this if you need other steps...

    def makeResult(self, grayFrame, flow):
        h, w = grayFrame.shape[:2]
        y, x = np.mgrid[self.step//2:h:self.step, self.step//2:w:self.step].reshape(2,-1)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

class DenseOpticalFlowByWarp(DenseOpticalFlow):
    def makeResult(self, grayFrame, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        return cv2.remap(grayFrame, flow, None, cv2.INTER_LINEAR)

class LucasKanadeOpticalFlow(IOpticalFlow):
    def __init__(self):
        # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 100,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        self.color = np.random.randint(0,255,(100,3))

    def set1stFrame(self, frame):
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask = None, **self.feature_params)
        # Create a mask image for drawing purposes
        self.mask = np.zeros_like(frame)

    def apply(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray,
                                               self.p0, None, **self.lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = self.p0[st==1]

        # draw the tracks
        for i, (new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            self.mask = cv2.line(self.mask, (a,b), (c,d), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (a,b), 5, self.color[i].tolist(), -1)
        img = cv2.add(frame, self.mask)

        # Now update the previous frame and previous points
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1,1,2)

        return img


def CreateOpticalFlow(type):
    '''Optical flow showcase factory, call by type as shown below'''
    def dense_by_hsv():
        return DenseOpticalFlowByHSV()
    def dense_by_lines():
        return DenseOpticalFlowByLines()
    def dense_by_warp():
        return DenseOpticalFlowByWarp()
    def lucas_kanade():
        return LucasKanadeOpticalFlow()
    return {
        'dense_hsv': dense_by_hsv,
        'dense_lines': dense_by_lines,
        'dense_warp': dense_by_warp,
        'lucas_kanade': lucas_kanade
    }.get(type, dense_by_lines)()
    
# class IOpticalFlow:
#     '''Interface of OpticalFlow classes'''
#     def set1stFrame(self, frame):
#         '''Set the starting frame'''
#         self.prev = frame

#     def apply(self, frame):
#         '''Apply and return result display image (expected to be new object)'''
#         result = frame.copy()
#         self.prev = frame
#         return result

# class DenseOpticalFlow(IOpticalFlow):
#     '''Abstract class for DenseOpticalFlow expressions'''
#     def set1stFrame(self, frame):
#         self.prev = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # self.hsv = np.zeros_like(frame)
#         # self.hsv[..., 1] = 255

    
#     def preprocess(img1_batch, img2_batch):
#         weights = Raft_Small_Weights.DEFAULT
#         transforms = weights.transforms()

#         img1_batch = F.resize(img1_batch, size=[128, 128], antialias=False)
#         img2_batch = F.resize(img2_batch, size=[128, 128], antialias=False)
#         return transforms(img1_batch, img2_batch)

#     def apply(self, frame):
#         next_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         next = next_img.copy()

#         # flow = cv2.calcOpticalFlowFarneback(self.prev, next, None,
#         #                                     0.5, 3, 15, 3, 5, 1.1, 0)
      
# # If you can, run this example on a GPU, it will be a lot faster.
#         device ="cpu"# "mps" if torch.backends.mps.is_available() else "cpu"

#         model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
#         model = model.eval()
#         prev = self.prev.copy()
#         prev = np.expand_dims(prev, axis=0)
#         next = np.expand_dims(next, axis=0)
        
#         #conver to float tensor

#         print(f"prev shape = {prev.shape}")
#         prev = torch.from_numpy(prev).float().to(device)
#         next = torch.from_numpy(next).float().to(device)
#         #last channel first

#         print(f"prev shape = {prev.shape}")
#         prev = prev.permute(0,3,1,2)
#         next = next.permute(0,3,1,2)
#         list_of_flows = model(prev, next)
#         print(f"type = {type(list_of_flows)}")
#         print(f"length = {len(list_of_flows)} = number of iterations of the model")
        
#         result = self.makeResult(next, list_of_flows[-1])#.cpu().numpy())
#         self.prev = next_img
#         return result

#     def makeResult(self, grayFrame, flow):
#         '''Replace this for each expression'''
#         from torchvision.utils import flow_to_image

#         flow_img = flow_to_image(flow)
#         flow_img = flow_img.permute(0,3,2,1).squeeze(0).numpy()
#         print(f"flow_img shape = {flow_img.shape}")

#         # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
#         # img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

#         return flow_img

# class DenseOpticalFlowByHSV(DenseOpticalFlow):
#     def makeResult(self, grayFrame, flow):
#         mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#         self.hsv[...,0] = ang*180/np.pi/2
#         self. hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#         return cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

# class DenseOpticalFlowByLines(DenseOpticalFlow):
#     def __init__(self):
#         self.step = 16 # configure this if you need other steps...

#     def makeResult(self, grayFrame, flow):
#         h, w = grayFrame.shape[:2]
#         y, x = np.mgrid[self.step//2:h:self.step, self.step//2:w:self.step].reshape(2,-1)
#         fx, fy = flow[y,x].T
#         lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#         lines = np.int32(lines + 0.5)
#         vis = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)
#         cv2.polylines(vis, lines, 0, (0, 255, 0))
#         for (x1, y1), (x2, y2) in lines:
#             cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
#         return vis

# class DenseOpticalFlowByWarp(DenseOpticalFlow):
#     def makeResult(self, grayFrame, flow):
#         h, w = flow.shape[:2]
#         flow = -flow
#         flow[:,:,0] += np.arange(w)
#         flow[:,:,1] += np.arange(h)[:,np.newaxis]
#         return cv2.remap(grayFrame, flow, None, cv2.INTER_LINEAR)

# class LucasKanadeOpticalFlow(IOpticalFlow):
#     def __init__(self):
#         # params for ShiTomasi corner detection
#         self.feature_params = dict( maxCorners = 100,
#                                     qualityLevel = 0.3,
#                                     minDistance = 7,
#                                     blockSize = 7 )

#         # Parameters for lucas kanade optical flow
#         self.lk_params = dict( winSize  = (15,15),
#                                maxLevel = 2,
#                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#         # Create some random colors
#         self.color = np.random.randint(0,255,(100,3))

#     def set1stFrame(self, frame):
#         self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask = None, **self.feature_params)
#         # Create a mask image for drawing purposes
#         self.mask = np.zeros_like(frame)

#     def apply(self, frame):
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # calculate optical flow
#         p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray,
#                                                self.p0, None, **self.lk_params)

#         # Select good points
#         good_new = p1[st==1]
#         good_old = self.p0[st==1]

#         # draw the tracks
#         for i, (new,old) in enumerate(zip(good_new,good_old)):
#             a,b = new.ravel()
#             c,d = old.ravel()
#             self.mask = cv2.line(self.mask, (a,b), (c,d), self.color[i].tolist(), 2)
#             frame = cv2.circle(frame, (a,b), 5, self.color[i].tolist(), -1)
#         img = cv2.add(frame, self.mask)

#         # Now update the previous frame and previous points
#         self.old_gray = frame_gray.copy()
#         self.p0 = good_new.reshape(-1,1,2)

#         return img


def CreateOpticalFlow(type):
    '''Optical flow showcase factory, call by type as shown below'''
    def dense_by_hsv():
        return DenseOpticalFlowByHSV()
    def dense_by_lines():
        return DenseOpticalFlowByLines()
    def dense_by_warp():
        return DenseOpticalFlowByWarp()
    def lucas_kanade():
        return LucasKanadeOpticalFlow()
    return {
        'dense_hsv': dense_by_hsv,
        'dense_lines': dense_by_lines,
        'dense_warp': dense_by_warp,
        'lucas_kanade': lucas_kanade
    }.get(type, dense_by_lines)()
    

def main():
    ## private routines
    def change(key, prevFrame):
        message, type = {
            ord('1'): ('==> Dense_by_hsv', 'dense_hsv'),
            ord('2'): ('==> Dense_by_lines', 'dense_lines'),
            ord('3'): ('==> Dense_by_warp', 'dense_warp'),
            ord('4'): ('==> Lucas-Kanade', 'lucas_kanade')
        }.get(key, ('==> Dense_by_hsv', 'dense_hsv'))
        print(message)
        of = CreateOpticalFlow(type)
        of.set1stFrame(prevFrame)
        return of
    
    def capture(vc):
        rval, frame = vc.read()
        if rval and flipImage:
            frame = cv2.flip(frame, 1)
        return (rval, frame)
        
    ## main starts here
    flipImage = True
    vc = cv2.VideoCapture(0)
    if not vc.isOpened():
        exit -1
            
    cv2.namedWindow("preview")
            
    ### try to get the first frame
    rval, frame = capture(vc)
    if rval:
        of = change('1', frame)

    ### main work
    while rval:
        rval, frame = capture(vc)

        ### do it
        img = of.apply(frame)
        cv2.imshow("preview", img)

        ### key operation
        key = cv2.waitKey(1)
        if key == 27:         # exit on ESC
            print('Closing...')
            break
        elif key == ord('s'):   # save
            cv2.imwrite('img_raw.png',frame)
            cv2.imwrite('img_w_flow.png',img)
            print("Saved raw frame as 'img_raw.png' and displayed as 'img_w_flow.png'")
        elif key == ord('f'):   # save
            flipImage = not flipImage
            print("Flip image: " + {True:"ON", False:"OFF"}.get(flipImage))
        elif ord('1') <= key and key <= ord('4'):
            of = change(key, frame)

    ## finish
    vc.release()
    cv2.destroyWindow("preview")


if __name__ == '__main__':
    print("s")
    main()