% read the image
I = imread('lenna.png');
figure(1)
imshow(I)
title("Original Image")

% rows and columns in the image
m = size(I, 1);
n = size(I, 2);

% storing the corresponding color plane

% red plane
R = I(:, :, 1);
R=uint8(R);
% green plane
G = I(:, :, 2);
G=uint8(G);
% blue plane
B =I(:, :, 3);
B=uint8(B);
% displaying the images
% figure, imshow(R);
% figure, imshow(G);
% figure, imshow(B);

% Converting red,blue,green planes into 1-D sequences
xr=reshape(R,1,[]);
xg=reshape(G,1,[]);
xb=reshape(B,1,[]);

% 3-D Logistic Map
% Create the two vectors xn and xnn, where x(i) and x(i+1) are stored respectively
xn=zeros(1,m*n);
xnn=zeros(1,m*n);
yn=zeros(1,m*n);
ynn=zeros(1,m*n);
zn=zeros(1,m*n);
znn=zeros(1,m*n);
% Give an initial value for 3-D logistic equations
xn(1)= 0.235; p=3.77;
yn(1)= 0.35; q=0.157;
zn(1)= 0.735; r=0.0125;

% Now iterate for m*n times - 
for i=1:(m*n)-1
 xnn(i)=(p*xn(i)*(1-xn(i)))+(q*yn(i)*yn(i)*xn(i))+(r*zn(i)*zn(i)*zn(i));
 xn(i+1)=xnn(i);
 ynn(i)=(p*yn(i)*(1-yn(i)))+(q*zn(i)*zn(i)*yn(i))+(r*xn(i)*xn(i)*xn(i));
 yn(i+1)=ynn(i);
 znn(i)=(p*zn(i)*(1-zn(i)))+(q*xn(i)*xn(i)*zn(i))+(r*yn(i)*yn(i)*yn(i));
 zn(i+1)=znn(i);
end
xnn(m*n)=(p*xn(m*n)*(1-xn(m*n)))+(q*yn(m*n)*yn(m*n)*xn(m*n))+(r*zn(m*n)*zn(m*n)*zn(m*n));
ynn(m*n)=(p*yn(m*n)*(1-yn(m*n)))+(q*zn(m*n)*zn(m*n)*yn(m*n))+(r*xn(m*n)*xn(m*n)*xn(m*n));
znn(m*n)=(p*zn(m*n)*(1-zn(m*n)))+(q*xn(m*n)*xn(m*n)*zn(m*n))+(r*yn(m*n)*yn(m*n)*yn(m*n));
% figure
% plot(xn,xnn)
% xlabel('xn')
% ylabel('xnn')
% grid;
%Quantification of each sequence - 
k=16; % secret key
xnq=zeros(1,m*n);
ynq=zeros(1,m*n);
znq=zeros(1,m*n);
for i=1:(m*n)
    xnq(i)=mod(round(xn(i)*(10^k)),256);
    ynq(i)=mod(round(yn(i)*(10^k)),256);
    znq(i)=mod(round(zn(i)*(10^k)),256);
end
% figure 
% plot(xn,'r')
% figure
% plot(xnq,'b')
% XOR operation
Xr = bitxor(xr,uint8(xnq));
Xg = bitxor(xg,uint8(ynq));
Xb = bitxor(xb,uint8(znq));

% Data Reconstruction

Xrmatrix=reshape(Xr,[m,n]);
Xgmatrix=reshape(Xg,[m,n]);
Xbmatrix=reshape(Xb,[m,n]);
% %Confusion
a = 1; b = 2;

t = 10; % Number of iterations
A = [1 a; b (1+(a*b))];
%Iterating through CAT map

 Xrcat = zeros(m,n);
 Xgcat = zeros(m,n);
 Xbcat = zeros(m,n);
    for i = 1:m         % loop through all the pixels
         for j = 1:n
                 newj=[i;j];
                 for f = 1:t                 
                   newj = catmap(newj,A,m);
                 end     
                 newj=reshape(newj,[1,2]);
                 newj=1.+newj; %since matlab doesn't take 0 index
                 Xrcat(newj(1),newj(2)) = Xrmatrix(i,j);
                 Xgcat(newj(1),newj(2)) = Xgmatrix(i,j);
                 Xbcat(newj(1),newj(2)) = Xbmatrix(i,j);
         end  
     end


% Cuboid1 = flipud(reshape(Cube3,M,M,3));
% 
% MatR = Cuboid1(:,:,1);
Xrcat = uint8(Xrcat);
% MatG = Cuboid1(:,:,2);
Xgcat = uint8(Xgcat);
% MatB = Cuboid1(:,:,3);
Xbcat = uint8(Xbcat);
Cipher = cat(3,Xrcat,Xgcat,Xbcat);
figure(2)
imshow(Cipher)
title("Encrypted Image")    

% Decryption
DMatR = Cipher(:,:,1);
DMatR=uint8(DMatR);
DMatG = Cipher(:,:,2);
DMatG=uint8(DMatG);
DMatB = Cipher(:,:,3);
DMatB=uint8(DMatB);
% Inverse catmap
DXrcat = zeros(m,n);
DXgcat = zeros(m,n);
DXbcat = zeros(m,n);
    for i = 1:m         % loop through all the pixels
         for j = 1:n
                 newj=[i;j];
                 for f = 1:t                 
                   newj = catmap(newj,inv(A),m);
                 end     
                 newj=reshape(newj,[1,2]);
                 newj=1.+newj; %since matlab doesn't take 0 index
                 DXrcat(newj(1),newj(2)) = DMatR(i,j);
                 DXgcat(newj(1),newj(2)) = DMatG(i,j);
                 DXbcat(newj(1),newj(2)) = DMatB(i,j);
         end  
     end
Dxr = uint8(reshape(DXrcat,1,[]));
Dxg = uint8(reshape(DXgcat,1,[]));
Dxb = uint8(reshape(DXbcat,1,[]));

% Logistic Equations
% XOR operation
DXr = bitxor(Dxr,uint8(xnq));
DXg = bitxor(Dxg,uint8(ynq));
DXb = bitxor(Dxb,uint8(znq));

DXrplane = uint8(reshape(DXr,m,n));
DXgplane = uint8(reshape(DXg,m,n));
DXbplane = uint8(reshape(DXb,m,n));

Decrypted = cat(3,DXrplane,DXgplane,DXbplane);
figure(3)
imshow(Decrypted)
title("Decrypted Image")   

 function Q = catmap(W,A,N)
     Q = mod((A*W),N);
 end

