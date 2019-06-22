clear;
close all;
scale=5;
nof=200;

%compute the average landmark postions
for nf=1:nof
    fid=fopen([num2str(nf),'.txt'],'r');
    lp=fscanf(fid,'%f');
    %将读入的数据变回二维数组的形式
    num2=reshape(mean(lp,2),2,round(size(lp,1)/2))';
    npo=size(num2,1);%特征点数量
    avern=mean(num2,1);%求取特征点的均值
    stdn=(std(num2(:,1),1)^2+std(num2(:,2),1)^2)^0.5;%求取特征点的方差
    for i=1:npo
        num02(i,1:2)=(num2(i,:)-avern)/stdn;%标准差，让landmarks符合正态分布
    end
    if nf==1
        num03=num02;
        num0=num02;
        avern0=avern;
        stdn0=stdn;
    else
        ang=mean(mod(atan(num0(:,2)./num0(:,1))-atan(num02(:,2)./num02(:,1))+pi/2,pi)-pi/2);
        num03=num02*[cos(ang) sin(ang);-sin(ang) cos(ang)];
    end
    nump(:,:,nf)=num03*stdn0+avern0;%对所有的landamarks做归一化处理
    fclose(fid);
end
lmpo=mean(nump,3);%根据第三维度即读取txt文档数量，求取平均特征值


%load average face mask image
avg_mask=imread('1_avg_mask.jpg');%载入脸孔轮廓图片

kp=avg_mask;
x001=size(kp,1);%图片第一维度H大小即X轴
y001=size(kp,2);%图片第二维度W大小即Y轴

%for循环遍历所有图片中的像素点，如果某个像素点为255将其周围横竖三个维度48个方格赋值成255.
for i=1:x001
    for j=1:y001
        if avg_mask(i,j,1)==255
            kp(max(i-3,1):min(i+3,x001),max(j-3,1):min(j+3:y001),1:3)=255;
        end
    end
end
avg_mask=kp;%然后在赋值回变量avg_mask数组

%记录数组H维度长度,并且以H维度自减计算所有像素点的像素平均值
%，如果RGB通道的均值大于127并且W维度上值都存在，则继续自减，去除白边
i=size(avg_mask,1);
% disp(i);
while sum(mean(avg_mask(i,:,:),3)>127)==size(avg_mask,2)
    i=i-1;
end
%记录数组W维度长度，并且以W维度自减计算所有像素点的像素平均值
%，如果RGB通道的均值大于127并且H维度上值都存在，则继续自减，去除白边
j=size(avg_mask,2);
while sum(mean(avg_mask(:,j,:),3)>127)==size(avg_mask,1)
    j=j-1;
end
%将整个头像轮廓保存
avg_mask=avg_mask(1:i,1:j,:);
%进行第二次降噪处理并放大保存
stai=1;
while sum(mean(avg_mask(stai,:,:),3)>127)==size(avg_mask,2)
    stai=stai+1;
end
staj=1;
while sum(mean(avg_mask(:,staj,:),3)>127)==size(avg_mask,1)
    staj=staj+1;
end
avg_mask2=avg_mask(stai:end,staj:end,:);%像素均值大大于127的点有29个

%Define shape and appearance descriptors. Shape are defined by landmark positions;
%Morph all faces to the average shape; use the average face mask to define
%the part of image going into the appearance descriptors.
mkdir('/home/fengchao/桌面/facedata/shape_appearance_PCs');
DST_PATH='/home/fengchao/桌面/facedata/shape_appearance_PCs/';
for nf=1:nof
    
    copyfile([num2str(nf),'.txt'],DST_PATH);%复制txt文档到另一路径
    filename=[num2str(nf),'.txt'];%打开txt文档
    
    filename2=[num2str(nf),'.jpg'];%打开文档同名图片
    
    filename3=[DST_PATH,num2str(nf),'.txt'];%打开另一路径下的同名文档
    
    kp=imread(filename2);%读取图片
    
    num=lmpo;%复制landmarks
    npo=size(num,1);%记录特征点数量
    avern0=mean(num,1);%求取特征点H维度均值
    stdn0=(std(num(:,1),1)^2+std(num(:,2))^2)^0.5;%求取特征点H和W维度均差
    
    for i=1:npo 
        %对68个特征点进行标准差标准化，符合标准正态分布 
        num0(i,1:2)=(num(i,:)-avern0)/stdn0;
    end
    
    fid=fopen(filename,'r');
    lp=fscanf(fid,'%f ');
    num2=reshape(mean(lp,2),2,round(size(lp,1)/2))';
    fclose(fid);
    
    avern=mean(num2,1);
    stdn=(std(num2(:,1),1)^2+std(num2(:,2))^2)^0.5;
    for i=1:npo
        num02(i,1:2)=(num2(i,:)-avern)/stdn;%对所有特征点的H和W维度进行标准化
    end
    %对数组中所有元素进行矩阵除法然后求反正切函数相减加上pi/2和PI求余再减去pi/2求平均数
    %对图片求面部特征坐标向量角度并对坐标进行旋转
    ang=mean(mod(atan(num0(:,2)./num0(:,1))-atan(num02(:,2)./num02(:,1))+pi/2,pi)-pi/2);
    num03=num02*[cos(ang) sin(ang);-sin(ang) cos(ang)];
    
    fid2 = fopen(filename3, 'w');
    
    for i=1:size(num03,1)
        fprintf(fid2,'%f %f\n',num03(i,1),num03(i,2));
    end
    %把上述求解写入另一路径txt文件中
    num3=num;
    num=num2;
    num2=num3;
    npo=size(num,1);
    
    for i=1:npo
        for j=1:npo
            if i==j%矩阵对角线为0
                L(i,j)=0;
            else
                dir=num(i,:)-num(j,:);%矩阵行与行做减法
                r2=sum(dir.*dir)/(scale^2);%对矩阵开方求和然后除以10
                L(i,j)=r2*log(r2)/log(10);%lg(x)=lg10(x)=>x=lg(x)/lg10
            end
        end
    end
    
    L(1:npo,npo+1)=1;%W维度69列中68个值为1
    L(1:npo,npo+2)=num(:,1);%W维度70列存放面部特征H维度所有值
    L(1:npo,npo+3)=num(:,2);%W维度71列存放面部特征W维度所有值
    L(npo+1,1:npo)=1;%H维度69列68个值为1
    L(npo+2,1:npo)=num(:,1)';%H维度70列存放面部特征H维度所有值
    L(npo+3,1:npo)=num(:,2)';%W维度71列存放面部特征W维度所有值
    Y(1:npo,1:2)=num2(:,1:2); %Y中存放所有图片landmarks的平均值
    Y(npo+1:npo+3,1:2)=0;%Y69-71列存放0值
    W=(L^(-1))*Y;%所有图片landmarks的均值除以L数组
    
    x0=size(kp,2);%文档同名图片W维度值
    y0=size(kp,1);%文档同名图片H维度值
    Xa(1:x0,1:y0)=0;%0数组空间
    Ya(1:x0,1:y0)=0;%0数组空间
    for i=1:x0
        Xa(i,:)=i;%Xa赋值1到1080 即图片W维度大小
    end
    for j=1:y0
        Ya(:,j)=j;%Ya赋值1到1920 即图片H维度大小
    end
    xo=W(npo+1,1)+Xa*W(npo+2,1)+Ya*W(npo+3,1);
    yo=W(npo+1,2)+Xa*W(npo+2,2)+Ya*W(npo+3,2);
    for i=1:npo
        
        r2=((Xa-num(i,1)).*(Xa-num(i,1))+(Ya-num(i,2)).*(Ya-num(i,2)))/(scale^2);
        xo=xo+W(i,1)*(r2.*log(r2))/log(10);
        yo=yo+W(i,2)*(r2.*log(r2))/log(10);
        
    end
    imfinal=[];
    for iu=1:3
        im12=[];
        im12(1:round(max(max(yo))),1:1:round(max(max(xo))))=0;%求出数组中的最大值并进行四舍五入
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
    imfinal=imfinal(1:size(avg_mask,1),1:size(avg_mask,2),:);%数组大小为面孔轮廓被去噪后的大小，数组内容为图片像素
    
    imfinal=imfinal(stai:end,staj:end,:);%对像素图片截取从29开始获取最终图像
    
    imfinal=mod(round(imfinal)-1,255)+1;
    h=0;
    %imfinal维度337*231*3
    for iu=1:3
        for i=1:size(imfinal,1)
            for j=1:size(imfinal,2)
                if mean(avg_mask2(i,j,:))<127
                    fprintf(fid2,'%d ',imfinal(i,j,iu));%最后将除噪特征点像素写入新txt文件
                    h=h+1;
                    if mod(h,20)==0
                        fprintf('\n');
                    end
                end
            end
        end
    end
    fclose(fid2);
end

%load shape and appearance descriptors and perform PCAs on these two descriptors

koo=[];
ko=[];
avg_maskh=impyramid(avg_mask2, 'reduce');%对mask图片进行金字塔处理，生成模糊度不同的多组相同图片，reduce代表分解——expand代表扩张
npo=size(lmpo,1);%获取landmaeks的数量
for nf=1:nof
%     fid=fopen(['shape_appearance_PCs',num2str(nf),'.txt'],'r');
    fid=fopen([DST_PATH,num2str(nf),'.txt'],'r');
    
    a1=fscanf(fid,'%f ');
    
    a3=a1(npo*2+1:end);%a3的数组大小为a1从68*2+1开始到a3最后
    
    a1=[a1(1:npo*2);mean(reshape(a3,length(a3)/3,3),2)];%前136行为landmarks的像素，后面用255填充
    %对a1进行z-score标准化，标准差标准化，经过处理的数据符合标准正态分布 
    a1(2*npo+1:length(a1))=(a1(2*npo+1:length(a1))-mean(a1(2*npo+1:length(a1))))/std(a1(2*npo+1:length(a1)))*40+75;

    koo(1:length(a1),nf)=a1;
    kp=[];
    kp(1:size(avg_mask2,1),1:size(avg_mask2,2))=85;
    h=npo*2;
    for i=1:size(avg_mask2,1)
        for j=1:size(avg_mask2,2)
            if mean(avg_mask2(i,j,:))<127
                h=h+1;
                kp(i,j)=a1(h);
            end
        end
    end
    kph=impyramid(kp, 'reduce');
    ko(1:npo*2,nf)=koo(1:npo*2,nf);
    h=npo*2;
    for i=1:size(avg_maskh,1)
        for j=1:size(avg_maskh,2)
            if mean(avg_maskh(i,j,:))<127
                h=h+1;
                ko(h,nf)=kph(i,j);
            end
        end
    end
    fclose(fid);
end

%PCA on shape and apperance vectors
shape=ko(1:npo*2,1:nof);%特征点形状
[COEFF1,SCORE1,latent1] = pca(shape');%COEFF1 136*2,SCORE1 3*2,latent1 2*1
% [COEFF1,SCORE1,latent1] = pca(shape');
%COEFF是原输入矩阵所对应的协方差阵的所有特征向量组成的矩阵，即变换矩阵或称投影矩阵，COEFF每列对应一个特征值的特征向量，列的排列顺序是按特征值的大小递减排序
%返回的SCORE是对主分的打分，也就是说原输入矩阵在主成分空间的表示。SCORE每行对应样本观测值，每列对应一个主成份(变量)，它的行和列的数目和X的行列数目相同
%返回的latent是enginvalue，它是输入矩阵所对应的协方差矩阵的特征值向量
sca2=1;
ko(1:npo*2,1:nof)=0;%面孔
[COEFF2,SCORE2,latent2] = pca(ko');%COEFF2 15249*2,SCORE2 3*2,latent2 2*1
%当维数p超过样本个数n的时候，用[...] = princomp(X,'econ')来计算，这样会显著提高计算速度
% [COEFF2,SCORE2,latent2] = pca(ko','econ');
s20=mean(koo,2);
h=0;
x002=size(avg_maskh,1);
y002=size(avg_maskh,2);
%b4 stores the average appearance image
b4=[];
b4(1:x002*2,1:y002*2)=85;
for i=1:size(avg_mask2,1)
    for j=1:size(avg_mask2,2)
        if mean(avg_mask2(i,j,:))<127
            h=h+1;
            if npo*2+h<=length(s20)
                b4(i,j)=s20(npo*2+h);
            end
        end
    end
end


%generate an aribitary face in the face space
mkdir('/home/fengchao/桌面/face_generation/general_template_face_generation/generate');
path='/home/fengchao/桌面/face_generation/general_template_face_generation/generate/';
%disp(size(latent1));%2,1
%disp(size(latent2));%2,1
for io=1:2000
    s2=mean(shape,2);
    %randomly sample points from the face space
    for ip=1:50
        if ip<=25
            b0(ip)=normrnd(0,latent1(ip)^(0.5));%出现下标越界，lantent维度为2*1
        else
            b0(ip)=normrnd(0,latent2(ip-25)^(0.5));
        end
    end
    % add the shape vectors to the average shape to compute the target shape
    for ik=1:25
        s2=s2+b0(ik)*COEFF1(1:2*npo,ik);
    end
    num02(:,1)=s2(1:2:npo*2);
    num02(:,2)=s2(2:2:npo*2);
    for tt=1:npo
        num2(tt,1:2)=num02(tt,:)*stdn0+avern0;
    end
    num=lmpo;
    num(:,1)=num(:,1)-staj+1;
    num(:,2)=num(:,2)-stai+1;
    L(1:npo,1:npo)=0;
    for i=1:npo
        for j=1:npo
            if i==j
                L(i,j)=0;
            else
                dir=num(i,:)-num(j,:);
                r2=sum(dir.*dir)/(scale^2);
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
    num3=num2;
    W=(L^(-1))*Y;
    % add the appearance vector to the average appearance to compute the target appearance
    s2=mean(ko,2);
    for ik=1:25
        s2=b0(ik+25)*double(COEFF2(:,ik))+s2;
    end
    s2=s2-mean(ko,2);
    
    
    h=0;
    b2=[];
    b2(1:size(avg_maskh,1),1:size(avg_mask,2))=0;
    for i=1:1:size(avg_maskh,1)
        for j=1:size(avg_maskh,2)
            if mean(avg_maskh(i,j,:))<127
                h=h+1;
                b2(i,j)=s2(npo*2+h);
            end
        end
    end
    
    s2=mean(koo,2);
    
    kp=[];
    kp(1:size(avg_mask2,1),1:size(avg_mask2,2))=85;
    h=npo*2;
    
    for i=1:size(avg_mask2,1)
        for j=1:size(avg_mask2,2)
            if mean(avg_mask2(i,j,:))<127
                h=h+1;
                kp(i,j)=s2(h);
            end
        end
    end
    b3= imresize(b2,size(b2)*2,'bilinear');
    b3=b3(1:size(b4,1),1:size(b4,2));
    b2=b3+b4;
    kp=b2;
    
    % morph the target appearance image to the target shape
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
    end
    im12=[];
    
    for x=1:x0
        
        for y=1:y0
            if abs(kp(y,x)-85)>0.0001
                im12(max(1,round(yo(x,y))),max(1,round(xo(x,y))))=kp(y,x);
            end
        end
    end
    
    for x=1:size(im12,1)
        
        for y=1:size(im12,2)
            if abs(im12(x,y))<0.0001
                im12(x,y)=85;
            end
        end
    end
    
    
    stax=1;
    st=0;
    endx=1;
    for i=1:size(im12,1)
        
        if mean(im12(i,:)==85)<1
            if st==0
                st=1;
                stax=i;
            end
            endx=i;
        end
    end
    
    
    stay=1;
    st=0;
    endy=1;
    for j=1:size(im12,2)
        
        if mean(im12(:,j)==85)<1
            if st==0
                st=1;
                stay=j;
            end
            endy=j;
        end
    end
    im12=im12(stax:endx,stay:endy);
    num3(:,1)=num3(:,1)-stay+1;
    num3(:,2)=num3(:,2)-stax+1;
    
    %fill in the gaps caused by morphing
    shift=[1 1; 1 -1; 1 0; 0 1; 0 -1; -1 1; -1 0; -1 -1];
    smwin=2;
    width=0.5;
    x01=size(im12,1);
    y01=size(im12,2);
    im5=im12;
    for i=1:x01
        
        for j=1:y01
            if abs(im12(i,j,:)-85)<0.0001
                ak(1:8)=0;
                for dir=1:8
                    ak(dir)=1;
                    for kk=1:6
                        ioo=i+shift(dir,1)*(kk);
                        joo=j+shift(dir,2)*(kk);
                        if ioo>0&&ioo<=x01&&joo>0&&joo<=y01
                            ak(dir)=ak(dir)*(abs(im12(ioo,joo,:)-85)<0.0001);
                        end
                    end
                end
                
                
                if sum(ak)>2
                else
                    
                    
                    ww=0;
                    ss=0;
                    for i1=max(i-smwin,1):min(i+smwin,x01)
                        for j1=max(j-smwin,1):min(j+smwin,y01)
                            w1=2.7183^(-((i1-i)^2+(j1-j)^2)/(width^2));
                            if i1<x01+1&&j1<y01+1&&i1>0&&j1>0&&abs(im5(i1,j1)-85)>0.0001
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
    end
    im5=im12;
    
    
    stax=1;
    st=0;
    endx=1;
    for i=1:x01
        
        if mean(abs(im5(i,:)-85)<1)<=0.98
            if st==0
                st=1;
                stax=i;
            end
            endx=i;
        end
    end
    
    
    stay=1;
    st=0;
    endy=1;
    for j=1:y01
        
        if mean(abs(im5(:,j)-85)<1)<=0.98
            if st==0
                st=1;
                stay=j;
            end
            endy=j;
        end
    end
    im5=im5(stax:endx,stay:endy);
    %output the image
    imwrite(im5/255,[path,'generate',num2str(io),'.tiff']);
    %output the 50-d feature vectors
    fid2 = fopen([path,'generate',num2str(io),'.txt'], 'w');
    for i=1:length(b0)
        fprintf(fid2,'%f ',b0(i));
    end
    fclose(fid2);
end


