clear;
close all;

scale=5;
 
%nof=number of faces
nof=200; 
%load landmark positions to compute the average position
for nf=1:nof
%landmark postions should saved in txt files, in the following format: x(1) y(1) x(2) y(2)... x(n) y(n)    
    %open text file 
    fid=fopen([num2str(nf),'.txt'],'r');
    %read opened text file into a stream
    %size of lp is (136, 1)
    lp=fscanf(fid,'%f ');
    %reshaping the file stream
    %using mean function to calculate the row mean value of landmarks,
    %if second parameter is 1 or empty, then it will calculate the mean value as the column;
    %if second parameter is 2, then it will calculate the mean value as the row.
    %using reshape function to reshape this array to 2 row and half columns of landmarks array rows. 
    %then use ' to transpose matrix.
    num2=reshape(lp,2,round(size(lp,1)/2))'; 
    %npo return the first demention of num2 which is 68.
    npo=size(num2,1);
    %calculating the mean value of landmarks as columns
    avern=mean(num2,1);%(2,1)
    %calculating the standard deviation of landmarks
    stdn=(std(num2(:,1),1)^2+std(num2(:,2))^2)^0.5;
    %normalize landmark postions to discount size and (average) location effect
    for i=1:npo %npo is the number of landmarks 
      num02(i,1:2)=(num2(i,:)-avern)/stdn;%normalized landmarks
    end
    %if this is first img landmarks then recording the normalized landmarks
    %and its mean value & srandard deviation
    if nf==1
        num03=num02;
        num0=num02;
        avern0=avern;%(2,1) landmarks mean value 
        stdn0=stdn;
    else
        %realign landmark positions to discont rotation effect
        ang=mean(mod(atan(num0(:,2)./num0(:,1))-atan(num02(:,2)./num02(:,1))+pi/2,pi)-pi/2);
        num03=num02*[cos(ang) sin(ang);-sin(ang) cos(ang)];
    end
    %nump stored all of the normalized landmarks and multiplies the
    %standard deviation then adding mean value
    nump(:,:,nf)=num03*stdn0+avern0;
    %closing file stream
    fclose(fid);
end

%caculating all of the handled landmarks' mean value by third dimention.
lmpo=mean(nump,3);%average landmarks

nf=1;
filename=[num2str(nf),'.txt'];
%original images should be saved as .jpgs
filename2=[num2str(nf),'.jpg'];
kp=imread(filename2);

num=lmpo;
npo=size(num,1);%num size is (58,2)

%load landmark positions for face 1.
fid=fopen(filename,'r');
lp=fscanf(fid,'%f');%(136,1)
fclose(fid);
num2=reshape(mean(lp,2),2,round(size(lp,1)/2))';%get mean values by row 
num3=num;%average landmarks
num=num2;%landmarks

num2=num3;%average landmarks
npo=size(num,1);%(68,2),68
%morph the face 1 to average shape
for i=1:npo %landmarks num
    for j=1:npo %landmarks num
        if i==j
            L(i,j)=0;%setting matrixs' diagonal line as 0
        else
            dir=num(i,:)-num(j,:);%the matrix minus all of its rows except same row.
            di=dir.*dir;%element multiplied element
            total=sum(di);%add by row
            sqrt=scale^2;
            r2=total/sqrt;%point multiply represented all of matrixs' elements to do the multiplication operation, then getting sum by column, divided to the sqrt of scale
            L(i,j)=r2*log(r2)/log(10);
        end
    end
end

L(1:npo,npo+1)=1;
L(1:npo,npo+2)=num(:,1);
L(1:npo,npo+3)=num(:,2);
L(npo+1,1:npo)=1;
L(npo+2,1:npo)=num(:,1)';
L(npo+3,1:npo)=num(:,2)';
Y(1:npo,1:2)=num2(:,1:2);
Y(npo+1:npo+3,1:2)=0;
W=(L^(-1))*Y;

x0=size(kp,2);
y0=size(kp,1);
Xa(1:x0,1:y0)=0;
Ya(1:x0,1:y0)=0;

for i=1:x0
    Xa(i,:)=i;
end
for j=1:y0
    Ya(:,j)=j;
end

xo=W(npo+1,1)+Xa*W(npo+2,1)+Ya*W(npo+3,1);
yo=W(npo+1,2)+Xa*W(npo+2,2)+Ya*W(npo+3,2);

for i=1:npo
    r2=((Xa-num(i,1)).*(Xa-num(i,1))+(Ya-num(i,2)).*(Ya-num(i,2)))/(scale^2);
    xo=xo+W(i,1)*(r2.*log(r2))/log(10);
    yo=yo+W(i,2)*(r2.*log(r2))/log(10);
    s=Xa-num(i,1)
end

%fill in the gaps caused by morphing
for iu=1:3
im12=[];
im12(1:round(max(max(yo))),1:1:round(max(max(xo))))=0;

for x=1:x0
    
    for y=1:y0
        im12(max(1,round(yo(x,y))),max(1,round(xo(x,y))))=kp(y,x,iu);
    end
end

shift=[1 1; 1 -1; 1 0; 0 1; 0 -1; -1 1; -1 0; -1 -1];
smwin=2;
x01=size(im12,1);
y01=size(im12,2);
im5=im12;


for i=1:x01

    for j=1:y01
         if im12(i,j)==0
            ww=0;
            ss=0;
            for i1=max(i-smwin,1):min(i+smwin,x01)
                for j1=max(j-smwin,1):min(j+smwin,y01)
                    w1=2.7183^(-((i1-i)^2+(j1-j)^2)/(2^2));
                    if i1<x01+1&&j1<y01+1&&i1>0&&j1>0&&im5(i1,j1)>0
                    ww=ww+w1;
                    ss=ss+w1*im5(i1,j1);
                    end
                end
            end
            if ww~=0
            im12(i,j)=ss/ww;
            end
 
        end
    end
end
imfinal(1:size(im12,1),1:size(im12,2),iu)=im12;
end
%output the otiginal image
imwrite(imfinal/255,[num2str(nf),'_avg.jpg']);
