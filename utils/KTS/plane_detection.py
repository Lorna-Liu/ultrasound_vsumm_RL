# import cv2
# import torch
# from skimage.transform import resize
# import json
# import collections
#
# def planedetection( frame):
#     json_filename = "./configs/config_sononet_8.json"
#     json_opts = json_file_to_pyobj(json_filename)
#     model = get_model(json_opts.model)
#     image = frame_loader(frame)
#     model.set_input(image)
#     model.net.eval()
#     with torch.no_grad():
#         model.forward(split='test')
#
#     model.logits = model.net.apply_argmax_softmax(model.prediction)
#     model.pred = model.logits.data.max(1)
#     scores = model.logits.data[0].cpu()
#     scores = scores.numpy()
#     # print(scores)
#     pr_lbls = model.pred[1].item()
#     print(pr_lbls)
#     return scores
#
# def frame_loader(frame, image_size=[224, 288]):
#     """load image, returns cuda tensor"""
#     image= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     im_array = resize(image, (int(image_size[0]), int(image_size[1] )), preserve_range=True)
#     image_T = torch.from_numpy(im_array)
#     image_T = image_T.type(torch.FloatTensor)
#     image_T = image_T.unsqueeze(0)
#     image_T = image_T.unsqueeze(0)
#     image_T_norm = image_T.sub(image_T.mean()).div(image_T.std())
#     return image_T_norm.cuda()
#
# def json_file_to_pyobj(filename):
#     def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
#     def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
#     return json2obj(open(filename).read())import collections