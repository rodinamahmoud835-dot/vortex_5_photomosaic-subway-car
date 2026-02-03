{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red195\green123\blue90;\red19\green20\blue21;\red174\green176\blue183;
\red89\green158\blue96;\red71\green149\blue242;\red38\green157\blue169;\red152\green54\blue29;\red117\green114\blue185;
}
{\*\expandedcolortbl;;\csgenericrgb\c76471\c48235\c35294;\csgenericrgb\c7451\c7843\c8235;\csgenericrgb\c68235\c69020\c71765;
\csgenericrgb\c34902\c61961\c37647;\csgenericrgb\c27843\c58431\c94902;\csgenericrgb\c14902\c61569\c66275;\csgenericrgb\c59608\c21176\c11373;\csgenericrgb\c45882\c44706\c72549;
}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f0\fs26 \cf2 \cb3 import \cf4 cv2\
\cf2 import \cf4 numpy \cf2 as \cf4 np\
\cf2 import \cf4 itertools\
\cf2 import \cf4 random\
stream1 = [cv2.imread(\cf5 "Resources/s1.jpg"\cf4 ),cv2.imread(\cf5 "Resources/s2.jpg"\cf4 ),cv2.imread(\cf5 "Resources/s3.jpg"\cf4 ),cv2.imread(\cf5 "Resources/s4.jpg"\cf4 ),cv2.imread(\cf5 "Resources/s5.jpg"\cf4 )]\
stream2 = [cv2.imread(\cf5 "Resources/s1.png"\cf4 ),cv2.imread(\cf5 "Resources/s2.png"\cf4 ),cv2.imread(\cf5 "Resources/s3.png"\cf4 ),cv2.imread(\cf5 "Resources/s4.png"\cf4 ),cv2.imread(\cf5 "Resources/s5.png"\cf4 )]\
\
\cf2 def \cf6 crop_white_board\cf4 (img):\
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)\
    mask = cv2.inRange(hsv, (\cf7 0\cf4 , \cf7 0\cf4 , \cf7 150\cf4 ), (\cf7 179\cf4 , \cf7 70\cf4 , \cf7 255\cf4 ))\
    kernel = np.ones((\cf7 7\cf4 , \cf7 7\cf4 ), np.uint8)\
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, \cf8 iterations\cf4 =\cf7 2\cf4 )\
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, \cf8 iterations\cf4 =\cf7 1\cf4 )\
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\
    \cf2 if \cf9 len\cf4 (contours) == \cf7 0\cf4 :\
        \cf2 return \cf4 img\
\
    biggest = \cf9 max\cf4 (contours, \cf8 key\cf4 =cv2.contourArea)\
    x, y, w, h = cv2.boundingRect(biggest)\
    var = \cf7 10\
    \cf4 x = \cf9 max\cf4 (\cf7 0\cf4 , x - var)\
    y = \cf9 max\cf4 (\cf7 0\cf4 , y - var)\
    w = \cf9 min\cf4 (img.shape[\cf7 1\cf4 ] - x, w + \cf7 2 \cf4 * var)\
    h = \cf9 min\cf4 (img.shape[\cf7 0\cf4 ] - y, h + \cf7 2 \cf4 * var)\
    \cf2 return \cf4 img[y:y+h, x:x+w]\
\
\cf2 def \cf6 get_edges\cf4 (img, t=\cf7 60\cf4 ):\
    h, w = img.shape[:\cf7 2\cf4 ]\
    x1 = \cf9 int\cf4 (w * \cf7 0.2\cf4 )\
    x2 = \cf9 int\cf4 (w * \cf7 0.8\cf4 )\
    y1 = \cf9 int\cf4 (h * \cf7 0.2\cf4 )\
    y2 = \cf9 int\cf4 (h * \cf7 0.8\cf4 )\
\
    top    = img[\cf7 0\cf4 :t, x1:x2]\
    bottom = img[h-t:h, x1:x2]\
    left   = img[y1:y2, \cf7 0\cf4 :t]\
    right  = img[y1:y2, w-t:w]\
    \cf2 return \cf4 top, bottom, left, right\
\
\cf2 def \cf6 edge_color_vector\cf4 (edge_bgr):\
    hsv = cv2.cvtColor(edge_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)\
    H = hsv[:, :, \cf7 0\cf4 ].reshape(-\cf7 1\cf4 )\
    S = hsv[:, :, \cf7 1\cf4 ].reshape(-\cf7 1\cf4 )\
    V = hsv[:, :, \cf7 2\cf4 ].reshape(-\cf7 1\cf4 )\
\
    n = \cf9 len\cf4 (S)\
    \cf2 if \cf4 n == \cf7 0\cf4 :\
        \cf2 return \cf4 np.array([\cf7 0.0\cf4 , \cf7 0.0\cf4 , \cf7 0.0\cf4 ], \cf8 dtype\cf4 =np.float32)\
\
    k = \cf9 int\cf4 (\cf7 0.12 \cf4 * n)\
    \cf2 if \cf4 k < \cf7 120\cf4 :\
        k = \cf9 min\cf4 (\cf7 120\cf4 , n)\
    idx = np.argpartition(S, n - k)[n - k:]\
    h = \cf9 float\cf4 (np.mean(H[idx]))\
    s = \cf9 float\cf4 (np.mean(S[idx]))\
    v = \cf9 float\cf4 (np.mean(V[idx]))\
    \cf2 return \cf4 np.array([h, s, v], \cf8 dtype\cf4 =np.float32)\
\
\cf2 def \cf6 hsv_dist\cf4 (a, b):\
    dh = \cf9 abs\cf4 (\cf9 float\cf4 (a[\cf7 0\cf4 ]) - \cf9 float\cf4 (b[\cf7 0\cf4 ]))\
    dh = \cf9 min\cf4 (dh, \cf7 180 \cf4 - dh)\
    ds = \cf9 abs\cf4 (\cf9 float\cf4 (a[\cf7 1\cf4 ]) - \cf9 float\cf4 (b[\cf7 1\cf4 ]))\
    dv = \cf9 abs\cf4 (\cf9 float\cf4 (a[\cf7 2\cf4 ]) - \cf9 float\cf4 (b[\cf7 2\cf4 ]))\
    \cf2 return \cf4 (\cf7 3.0 \cf4 * dh) + (\cf7 1.0 \cf4 * ds) + (\cf7 0.2 \cf4 * dv)\
\
\cf2 def \cf6 rotate_image\cf4 (img, rot):\
    \cf2 if \cf4 rot == \cf7 0\cf4 :\
        \cf2 return \cf4 img\
    \cf2 return \cf4 cv2.rotate(img, cv2.ROTATE_180)\
\
\cf2 def \cf6 precompute\cf4 (imgs, t=\cf7 60\cf4 , rots=(\cf7 0\cf4 , \cf7 180\cf4 )):\
    crops = \{\}\
    vecs = \{\}\
\
    \cf2 for \cf4 i \cf2 in \cf9 range\cf4 (\cf9 len\cf4 (imgs)):\
        \cf2 for \cf4 r \cf2 in \cf4 rots:\
            imr = rotate_image(imgs[i], r)\
            crop = crop_white_board(imr)\
            crops[(i, r)] = crop\
\
            top, bottom, left, right = get_edges(crop, t)\
            vecs[(i, r)] = \{\
                \cf5 "top"\cf4 : edge_color_vector(top),\
                \cf5 "bottom"\cf4 : edge_color_vector(bottom),\
                \cf5 "left"\cf4 : edge_color_vector(left),\
                \cf5 "right"\cf4 : edge_color_vector(right),\
            \}\
    \cf2 return \cf4 crops, vecs\
\
\cf2 def \cf6 kmeans\cf4 (points, k=\cf7 8\cf4 , iters=\cf7 25\cf4 , seed=\cf7 0\cf4 ):\
    random.seed(seed)\
    pts = np.array(points, \cf8 dtype\cf4 =np.float32)\
    idxs = \cf9 list\cf4 (\cf9 range\cf4 (\cf9 len\cf4 (pts)))\
    random.shuffle(idxs)\
    centers = pts[idxs[:k]].copy()\
    \cf2 for \cf4 _ \cf2 in \cf9 range\cf4 (iters):\
        labels = []\
        \cf2 for \cf4 p \cf2 in \cf4 pts:\
            dmin = \cf2 None\
            \cf4 best = \cf7 0\
            \cf2 for \cf4 ci \cf2 in \cf9 range\cf4 (k):\
                d = hsv_dist(p, centers[ci])\
                \cf2 if \cf4 dmin \cf2 is None or \cf4 d < dmin:\
                    dmin = d\
                    best = ci\
            labels.append(best)\
        labels = np.array(labels, \cf8 dtype\cf4 =np.int32)\
\
        new_centers = centers.copy()\
        \cf2 for \cf4 ci \cf2 in \cf9 range\cf4 (k):\
            mask = (labels == ci)\
            \cf2 if \cf4 np.count_nonzero(mask) > \cf7 0\cf4 :\
                new_centers[ci] = np.mean(pts[mask], \cf8 axis\cf4 =\cf7 0\cf4 )\
        centers = new_centers\
    \cf2 return \cf4 centers, labels\
\
\cf2 def \cf6 build_color_ids\cf4 (vecs, rots=(\cf7 0\cf4 , \cf7 180\cf4 ), k=\cf7 8\cf4 ):\
    keys = []\
    points = []\
    \cf2 for \cf4 (i, r), d \cf2 in \cf4 vecs.items():\
        \cf2 if \cf4 r \cf2 not in \cf4 rots:\
            \cf2 continue\
        for \cf4 e \cf2 in \cf4 [\cf5 "top"\cf4 , \cf5 "bottom"\cf4 , \cf5 "left"\cf4 , \cf5 "right"\cf4 ]:\
            keys.append((i, r, e))\
            points.append(d[e])\
    centers, labels = kmeans(points, \cf8 k\cf4 =k, \cf8 iters\cf4 =\cf7 30\cf4 , \cf8 seed\cf4 =\cf7 0\cf4 )\
    id_of = \{\}\
    \cf2 for \cf4 idx, key \cf2 in \cf9 enumerate\cf4 (keys):\
        id_of[key] = \cf9 int\cf4 (labels[idx])\
    \cf2 return \cf4 id_of, centers\
\
\cf2 def \cf6 solve_strict\cf4 (imgs, vecs, id_of, rots=(\cf7 0\cf4 , \cf7 180\cf4 )):\
    ids = \cf9 list\cf4 (\cf9 range\cf4 (\cf9 len\cf4 (imgs)))\
    best = \cf2 None\
    for \cf4 top_id \cf2 in \cf4 ids:\
        sides = [i \cf2 for \cf4 i \cf2 in \cf4 ids \cf2 if \cf4 i != top_id]\
        \cf2 for \cf4 perm \cf2 in \cf4 itertools.permutations(sides, \cf7 4\cf4 ):\
            \cf2 for \cf4 side_rots \cf2 in \cf4 itertools.product(rots, \cf8 repeat\cf4 =\cf7 4\cf4 ):\
                ok = \cf2 True\
                \cf4 score = \cf7 0.0\
                \cf2 for \cf4 k \cf2 in \cf9 range\cf4 (\cf7 3\cf4 ):\
                    a_id = id_of[(perm[k], side_rots[k], \cf5 "right"\cf4 )]\
                    b_id = id_of[(perm[k+\cf7 1\cf4 ], side_rots[k+\cf7 1\cf4 ], \cf5 "left"\cf4 )]\
                    \cf2 if \cf4 a_id != b_id:\
                        ok = \cf2 False\
                    \cf4 score += hsv_dist(vecs[(perm[k], side_rots[k])][\cf5 "right"\cf4 ],\
                                      vecs[(perm[k+\cf7 1\cf4 ], side_rots[k+\cf7 1\cf4 ])][\cf5 "left"\cf4 ])\
                \cf2 if not \cf4 ok:\
                    \cf2 continue\
                for \cf4 top_rot \cf2 in \cf4 rots:\
                    top_edge_id = id_of[(top_id, top_rot, \cf5 "bottom"\cf4 )]\
                    \cf2 for \cf4 attach_index \cf2 in \cf9 range\cf4 (\cf7 4\cf4 ):\
                        side_top_id = id_of[(perm[attach_index], side_rots[attach_index], \cf5 "top"\cf4 )]\
                        \cf2 if \cf4 top_edge_id != side_top_id:\
                            \cf2 continue\
                        \cf4 score2 = score + hsv_dist(vecs[(top_id, top_rot)][\cf5 "bottom"\cf4 ],\
                                                  vecs[(perm[attach_index], side_rots[attach_index])][\cf5 "top"\cf4 ])\
                        \cf2 if \cf4 best \cf2 is None or \cf4 score2 < best[\cf7 0\cf4 ]:\
                            best = (score2, perm, side_rots, top_id, top_rot, attach_index)\
\
    \cf2 return \cf4 best\
\
\cf2 def \cf6 resize_to_height\cf4 (img, target_h):\
    h, w = img.shape[:\cf7 2\cf4 ]\
    \cf2 if \cf4 h == target_h:\
        \cf2 return \cf4 img\
    new_w = \cf9 int\cf4 (w * (target_h / \cf9 float\cf4 (h)))\
    \cf2 return \cf4 cv2.resize(img, (new_w, target_h), \cf8 interpolation\cf4 =cv2.INTER_AREA)\
\
\cf2 def \cf6 build_mosaic\cf4 (crops, solution):\
    score, side_ids, side_rots, top_id, top_rot, attach_index = solution\
\
    sides = [crops[(side_ids[i], side_rots[i])] \cf2 for \cf4 i \cf2 in \cf9 range\cf4 (\cf7 4\cf4 )]\
    side_h = \cf9 min\cf4 ([im.shape[\cf7 0\cf4 ] \cf2 for \cf4 im \cf2 in \cf4 sides])\
    sides = [resize_to_height(im, side_h) \cf2 for \cf4 im \cf2 in \cf4 sides]\
    row = cv2.hconcat(sides)\
\
    top_img = crops[(top_id, top_rot)]\
\
    attach_w = sides[attach_index].shape[\cf7 1\cf4 ]\
    th, tw = top_img.shape[:\cf7 2\cf4 ]\
    new_h = \cf9 int\cf4 (th * (attach_w / \cf9 float\cf4 (tw)))\
    \cf2 if \cf4 new_h <= \cf7 0\cf4 :\
        new_h = th\
    top_resized = cv2.resize(top_img, (attach_w, new_h), \cf8 interpolation\cf4 =cv2.INTER_AREA)\
\
    x_off = \cf9 sum\cf4 ([sides[i].shape[\cf7 1\cf4 ] \cf2 for \cf4 i \cf2 in \cf9 range\cf4 (attach_index)])\
\
    row_h, row_w = row.shape[:\cf7 2\cf4 ]\
    top_h, top_w = top_resized.shape[:\cf7 2\cf4 ]\
\
    canvas = np.ones((row_h + top_h, row_w, \cf7 3\cf4 ), \cf8 dtype\cf4 =np.uint8) * \cf7 255\
    \cf4 canvas[\cf7 0\cf4 :top_h, x_off:x_off+top_w] = top_resized\
    canvas[top_h:top_h+row_h, \cf7 0\cf4 :row_w] = row\
    \cf2 return \cf4 canvas\
\
imgs = stream2\
\
\
T = \cf7 60\
\cf4 ROTS = (\cf7 0\cf4 , \cf7 180\cf4 )\
K_COLORS = \cf7 8\
\cf4 crops, vecs = precompute(imgs, \cf8 t\cf4 =T, \cf8 rots\cf4 =ROTS)\
id_of, centers = build_color_ids(vecs, \cf8 rots\cf4 =ROTS, \cf8 k\cf4 =K_COLORS)\
solution = solve_strict(imgs, vecs, id_of, \cf8 rots\cf4 =ROTS)\
final_img = build_mosaic(crops, solution)\
cv2.imwrite(\cf5 "photomosaic_result.png"\cf4 , final_img)\
cv2.imshow(\cf5 "Photomosaic Result"\cf4 , final_img)\
cv2.waitKey(\cf7 0\cf4 )\
\
\
}