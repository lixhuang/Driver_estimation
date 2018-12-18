load('filtered_train_data.mat');
N = length(train_data_len);
%x=[];
%x_len = train_data_len;
x = {};

for i =1:length(train_data)
    origin = train_data(i).bv(1,3);
    xf = train_data(i).fv(:,3) - origin;
    vf = train_data(i).fv(:,4);
    xc = train_data(i).cv(:,3) - origin;
    vxc = train_data(i).cv(:,4);
    yc = train_data(i).cv(:,2);
    vyc = [0;diff(train_data(i).cv(:,4))/0.1];
    xb = train_data(i).bv(:,3) - origin;
    vb = train_data(i).bv(:,4);
    %% dowsampling

    
    x{i} = [xf,vf,xc,vxc,yc,vyc,xb,vb]';
end
save('proc_data.mat', 'x');