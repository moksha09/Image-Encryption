% read the image
I = imread('lenna.png');

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

% Concerting red,blue,green planes into 1-D sequences
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
N=ceil((m*n*3)^(1/3));
Xrgb = [Xr Xg Xb];   % Combining three sequence into a single sequence
t=N*N*N-size(Xrgb,2); % Size of remaining no. of elements required to fill the cube
Ran = randi([0,255],1,t); % Creating random numbers for rest of the cude
Xrgb = [Xrgb Ran]; %Adding the random numbers at the end of the single sequence
Cube1=permute(reshape(Xrgb,N,N,N),[1,2,3]); % Forming the Cube

%Confusion
t = 10; % Number of iterations
AB = randi([1,N],1,6); % Assigning a,b secret keys to random integers
ax=AB(1);ay=AB(2); az=AB(3); bx=AB(4); by=AB(5); bz=AB(6);
% Making marix A
A11 = (ax*az*by)+1; A12 = az; A13 = ay+(ax*az)+(ax*ay*az*by);
A21 = bz+(ax*by)+(ax*az*by*bz); A22 = (az*bz)+1; A23 = (ay*az)+(ax*ay*az*by*bz)+(ax*az*bz)+(ax*ay*by)+ax;
A31 = (ax*bx*by)+by; A32 = bx; A33 = (ax*ay*bx*by)+(ax*bx)+(ay*by)+1;
A = [A11 A12 A13; A21 A22 A23; A31 A32 A33];
%Iterating through CAT map
Cube2 = zeros(N,N,N);
    for k = 1:N
        for i = 1:N         % loop through all the pixels
            for j = 1:N
                newj = mod((A*[i;j;k]),N);    
                newj=reshape(newj,[1,3]);
                newj=1.+newj; %since matlab doesn't take 0 index
                Cube2(newj(1),newj(2),newj(3)) = Cube1(i,j,k);
            end  
        end
    end
M = ceil(((N*N*N)/3)^(1/2))

T=(M*M*3)-(N*N*N); % Size of remaining no. of elements required to fill the cube
Ran = randi([0,255],1,T);
Cube3 = flipud(Cube2);
Cube3 = reshape(Cube3,1,[]);
Cube3 = [ Cube3 Ran ];
Cuboid1 = flipud(reshape(Cube3,M,M,3));

MatR = Cuboid1(:,:,1);
MatR = uint8(MatR);
MatG = Cuboid1(:,:,2);
MatG = uint8(MatG);
MatB = Cuboid1(:,:,3);
MatB = uint8(MatB);
Cipher = cat(3,MatR,MatG,MatB);
%X = uint8(X);   % this may have to be adjusted depending on the type of image
figure(2)
imshow(Cipher)
title("Encrypted Image")
