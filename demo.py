from torch.autograd import Variable
import torch
import cv2
import numpy as np

from layers.proposal import ProposalLayer
from layers.roi_align import RoIAlign
from net import BackBone, RPN, FasterRCNN
from utils.results import postprocess_output
from utils.blob import im_list_to_blob, prep_im_for_blob
from utils import vis

############# define some configurations: ################

# Number of top scoring boxes to keep before apply NMS to RPN proposals
RPN_PRE_NMS_TOP_N = 6000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
RPN_POST_NMS_TOP_N = 1000
# NMS threshold used on RPN proposals
RPN_NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
RPN_MIN_SIZE = 0
# Size of the pooled region after RoI pooling
POOLING_SIZE = 14

TEST_NMS = 0.5
TEST_MAX_SIZE = 1333
PIXEL_MEANS = np.array([122.7717, 115.9465, 102.9801])

anchor_sizes, anchor_ratios = [32, 64, 128, 256, 512], [0.5, 1, 2]
feat_stride = 16

############ prepare an image #################

x = cv2.imread('samples/15673749081_767a7fa63a_k.jpg')[:, :, ::-1]

blobs, im_scales = prep_im_for_blob(x, PIXEL_MEANS, target_sizes=(800,), max_size=TEST_MAX_SIZE)
blobs = im_list_to_blob(blobs)
img = Variable(torch.from_numpy(blobs))

im_info = torch.from_numpy(np.array([[blobs.shape[2], blobs.shape[3], im_scales[0]]], dtype=np.float32))
im_size, im_scale = [blobs.shape[2], blobs.shape[3]], im_scales[0]

########## make the modules ####################

backbone = BackBone()
proposal_layer = ProposalLayer(feat_stride, anchor_sizes, anchor_ratios,
                               RPN_PRE_NMS_TOP_N, RPN_POST_NMS_TOP_N, RPN_NMS_THRESH, RPN_MIN_SIZE)
roi_align = RoIAlign(POOLING_SIZE, POOLING_SIZE, spatial_scale=1./16.)

rpn = RPN(1024, 1024, 15, proposal_layer)
frcnn = FasterRCNN(rpn, backbone, roi_align, 81)
frcnn.load_pretrained_weights('model_final.pkl', 'resnet50_mapping.npy')

############ feedforward pass with postprocessing ################

frcnn = frcnn.cuda()
frcnn.eval()
img = img.cuda()

output = frcnn(img, im_size[0], im_size[1], im_scale)
class_scores, bbox_deltas, rois = output['cls_prob'], output['bbox_pred'], output['rois']

scores_final, boxes_final, boxes_per_class = postprocess_output(rois, im_scale, im_size, class_scores, bbox_deltas,
                                                                bbox_reg_weights=(10.0, 10.0, 5.0, 5.0))

########## visualize ###############

vis.vis_one_image(
    x,  # BGR -> RGB for visualization
    'output',
    'samples/',
    boxes_per_class,
    dataset=None,
    box_alpha=0.3,
    show_class=True,
    thresh=0.7,
    ext='jpg'
)
