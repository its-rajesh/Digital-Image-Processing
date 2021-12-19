image = imread("/Users/rajeshr/Desktop/peppers.png");
gray_image=rgb2gray(image);

bit_6=gray_image*(63/255);
bit_6_converted=im2uint8(bit_6);
subplot(3,1,1);
imshow(bit_6_converted, []);
title("6 bit");

bit_4=gray_image*(15/255);
bit_4_converted=im2uint8(bit_4);
subplot(3,1,2);
imshow(bit_4_converted, []);
title("4 bit");

bit_2=gray_image*(4/255);
bit_2_converted=im2uint8(bit_2);
subplot(3,1,3);
imshow(bit_2_converted, []);
title("2 bit");