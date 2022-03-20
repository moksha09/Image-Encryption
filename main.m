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
xn(1)= 0.2350; p=3.77;
yn(1)= 0.3500; q=0.0157;
zn(1)= 0.7350; r=0.0125;

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
% figure (9)
% plot(xn)
% hold on 
% plot(yn)
% hold on
% plot(zn)
% hold off


%Quantification of each sequence - 
k=5; % secret key
xnq=zeros(1,m*n);
ynq=zeros(1,m*n);
znq=zeros(1,m*n);
for i=1:(m*n)
    xnq(i)=mod(round(xn(i)*(10^k)),256);
    ynq(i)=mod(round(yn(i)*(10^k)),256);
    znq(i)=mod(round(zn(i)*(10^k)),256);
end
% figure (4)
% plot(xn,'r')
% figure (5)
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
    for i = 0:m-1         % loop through all the pixels
         for j = 0:n-1
                 newj=[i;j];
                 for f = 1:t                 
                   newj = catmap(newj,A,m);
                 end     
                 newj=reshape(newj,[1,2]);
                 newj=1.+newj; %since matlab doesn't take 0 index
                 Xrcat(newj(1),newj(2)) = Xrmatrix(i+1,j+1);
                 Xgcat(newj(1),newj(2)) = Xgmatrix(i+1,j+1);
                 Xbcat(newj(1),newj(2)) = Xbmatrix(i+1,j+1);
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
    for i = 0:m-1         % loop through all the pixels
         for j = 0:n-1
                 newj=[i;j];
                 for f = 1:t                 
                   newj = catmap(newj,inv(A),m);
                 end     
                 newj=reshape(newj,[1,2]);
                 newj=1.+newj; %since matlab doesn't take 0 index
                 DXrcat(newj(1),newj(2)) = DMatR(i+1,j+1);
                 DXgcat(newj(1),newj(2)) = DMatG(i+1,j+1);
                 DXbcat(newj(1),newj(2)) = DMatB(i+1,j+1);
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

% Key Sensitivity


% Histogram Analysis
figure(4)
subplot(1,2,1)
imhist(I)
title('Histogram of Original Image')
subplot(1,2,2)
imhist(Cipher)
title('Histogram of Cipher Image')

% Correlation Analyis
AdjancyCorrPixelRand(I,Cipher)

% Information Entropy
disp("Lenna Cipher image Entropy - ")
entropy(Cipher)


function Q = catmap(W,A,N)
     Q = mod((A*W),N);
end

function CC=AdjancyCorrPixelRand(Orignal,Encrypted) 
    Orignal=double(Orignal);
    Encrypted=double(Encrypted);
    %%
    k = 100;
    [m,n] = size(Orignal);    %// works with 3- and 1-channel images
    m=m-1;
    n=n-1;
    s = randsample(m*n, k);        %// get k random indices
    [X, Y] = ind2sub([m, n], s);   %// convert indices to x,y coordinates
    %% horizontal
    hxO = Orignal(X,Y); 
    hyO = Orignal(X,Y+1); 
    Ho_xy = corrcoef(hxO,hyO);
    
    figure('Name','Correlation Coefficient'), 
    subplot(3,2,1), %title('horizontal Original')
    scatter(hxO(:),hyO(:),'.')
    axis([0 255 0 255]) 
    box on
    xlabel('Pixel value on location (x,y)') 
    ylabel('Pixel value on location (x,y+1)') 
    
    hxE = Encrypted(X,Y); 
    hyE = Encrypted(X,Y+1); 
    He_xy = corrcoef(hxE,hyE);
    subplot(3,2,2), %title('horizontal Encrypted')
    scatter(hxE(:),hyE(:),'.')
    axis([0 255 0 255])
    box on
    xlabel('Pixel value on location (x,y)') 
    ylabel('Pixel value on location (x,y+1)') 
    
    CC(1,1)=Ho_xy(1,2);
    CC(1,2)=He_xy(1,2);
    
    
    %% vertical 
    vxO = Orignal(X,Y); 
    vyO = Orignal(X+1,Y); 
    Vo_xy = corrcoef(vxO,vyO);
    subplot(3,2,3), %title('vertical Original')
    scatter(vxO(:),vyO(:),'.')
    axis([0 255 0 255]) 
    box on
    xlabel('Pixel value on location (x,y)') 
    ylabel('Pixel value on location (x+1,y)') 
    
    vxE = Encrypted(X,Y); 
    vyE = Encrypted(X+1,Y); 
    Ve_xy = corrcoef(vxE,vyE);
    subplot(3,2,4), %title('vertical Encrypted')
    scatter(vxE(:),vyE(:),'.')
    axis([0 255 0 255])
    box on
    xlabel('Pixel value on location (x,y)') 
    ylabel('Pixel value on location (x+1,y)') 
    
    CC(2,1)=Vo_xy(1,2);
    CC(2,2)=Ve_xy(1,2);
    
    %% diagonal 
    dxO = Orignal(X,Y); 
    dyO = Orignal(X+1,Y+1); 
    Do_xy = corrcoef(dxO,dyO);
    subplot(3,2,5), %title('Diagonal Original')
    scatter(dxO(:),dyO(:),'.')
    axis([0 255 0 255]) 
    box on
    xlabel('Pixel value on location (x,y)') 
    ylabel('Pixel value on location (x+1,y+1)') 
    
    dxE = Encrypted(X,Y); 
    dyE = Encrypted(X+1,Y+1); 
    De_xy = corrcoef(dxE,dyE);
    subplot(3,2,6), %title('Diagonal Encrypted')
    scatter(dxE(:),dyE(:),'.')
    axis([0 255 0 255])
    box on
    xlabel('Pixel value on location (x,y)') 
    ylabel('Pixel value on location (x+1,y+1)') 
    
    CC(3,1)=Do_xy(1,2);
    CC(3,2)=De_xy(1,2);
    
    %% Defaults for this blog post
%     width = 9;     % Width in inches
%     height = 12;    % Height in inches
%     alw = 0.75;    % AxesLineWidth
%     fsz = 5;      % Fontsize
%     lw = 1.5;      % LineWidth
%     msz = 2;       % MarkerSize
%     pos = get(gcf, 'Position');
%     set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
%     %set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
%     %set(gca, 'FontWeight','bold');
%     set(gcf,'InvertHardcopy','on');
%     set(gcf,'PaperUnits', 'inches');
%     papersize = get(gcf, 'PaperSize');
%     left = (papersize(1)- width)/2;
%     bottom = (papersize(2)- height)/2;
%     myfiguresize = [left, bottom, width, height];
%     set(gcf,'PaperPosition', myfiguresize);
% 
%     % Save the file as PNG
%     print('FigCC','-dtiff','-r300');
    
end

function [npcr, uaci] = NPCR_UACI(ChiperImg,ChiperImg1bit)
    f1 = double(ChiperImg);
    f2 = double(ChiperImg1bit);
    [M, N] = size(f1);
    %% NPCR
    
    d = 0.000000;
    for i = 1 : M
        for j = 1 : N
            if f1(i, j) ~= f2(i, j)         
               d = d + 1;
            end
        end
    end
    npcr = d / (M * N);

    %% UACI
    c = 0.000000;
    for i = 1 : M * N
         c = c + abs( double( f1(i)) - double( f2(i)));
    end
    uaci = c / (255 * M * N);
end

