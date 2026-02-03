{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 **Description:**  \
This task autonomously creates a photomosaic from multiple image pieces of a submerged subway car.\
Each image piece is cropped, its edges are analyzed, and the correct arrangement is found by matching\
edge colors and orientations. The final result is a single combined photomosaic image.\
\
**Method Overview:**\
- Load multiple image pieces (stream images)\
- Crop the white board area to remove background\
- Extract top, bottom, left, and right edges from each piece\
- Represent each edge using HSV color features\
- Cluster edge colors using K-means to obtain discrete color IDs\
- Search for a valid arrangement where adjacent edges match\
- Assemble and display the final photomosaic}