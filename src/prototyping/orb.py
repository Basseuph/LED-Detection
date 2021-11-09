import numpy as np
import cv2
from src.prototyping.corner import corner_filter

# Read the query image as query_img
# and train image This query image
# is what you need to find in train image
# Save it in the same directory
# with the name image.jpg


query_img = corner_filter('./resources/PiWithCables/distant.jpg')
train_img = corner_filter('./resources/PiWithCables/close.jpg')

# Initialize the ORB detector algorithm
orb = cv2.ORB_create()

# Now detect the keypoints and compute
# the descriptors for the query image
# and train image
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img, None)

# Initialize the Matcher for matching
# the keypoints and then match the
# keypoints
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)

# draw the matches to the final image
# containing both the images the drawMatches()
# function takes both images and keypoints
# and outputs the matched query image with
# its train image
final_img = cv2.drawMatches(query_img, queryKeypoints,
                            train_img, trainKeypoints, matches[:20], None)

final_img = cv2.resize(final_img, (1000, 650))

# Show the final image
cv2.imshow("Matches", final_img)
cv2.waitKey(30000)