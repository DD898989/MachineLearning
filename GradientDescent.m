clear all
clc
 
x = [  
    12.0413,    36.1239,    134.0596,   164.5642,   156.5367,...
    172.5917,   212.7294,   302.6376,   520.9862,   620.5275    ];
y = [  
    39.6501,    104.9563,   160.9329,   198.2507,   319.5335,...
    342.8571,   422.1574,   566.7638,   1.3085e+03, 1.5917e+03  ];
 

hold on
xlabel('w')
ylabel('b')
zlabel('Err')
syms symsW
syms symsB


plot3( 2.6267,-107.2144,ErrSum(2.6267,-107.2144,x,y) , '.r', 'MarkerSize',20)%theory minimum point


[w,b] = meshgrid(-6:1:6,-200:5:100);
Err = ErrSum(w,b,x,y);
%surf(w,b,Err) %3D surf

 
 currW = -4.6267; %start point
 currB = 80.2144; %start point
 plot3(currW,currB,ErrSum(currW,currB,x,y),'md') %start point
% ErrSumPW = (diff(ErrSum( symsW, currW, x,y),symsW));%not using default gradien function
% ErrSumPB = (diff(ErrSum( currB, symsB, x,y),symsB));%not using default gradien function
 
 num = 100;
 c = [currW currB ,ErrSum(currW,currB,x,y)];
 grad_f=eval(gradient(ErrSum(symsW,symsB,x,y)));
 for i=1:num
        %nextW = (currW-0.00000101*subs(ErrSumPW,symsW,currW)); %not using default gradien function
        %nextB = (currB-0.00000101*subs(ErrSumPB,symsB,currB)); %not using default gradien function
        %c = [c;      nextW  nextB  ErrSum(nextW,nextB,x,y)];   %not using default gradien function
       
        delta = eval(subs(grad_f, {symsW, symsB}, {currW, currB}));
        next = [currW currB]-0.00003*delta;
        nextW=next(1);
        nextB=next(2);
        c = [c;      (nextW)  (nextB)  ErrSum(nextW,nextB,x,y)];
       
        currW=nextW; 
        currB=nextB;
 end
 colors = jet(length(c));
 plot3(c(:,1),c(:,2),c(:,3))
 for i=1:length(c)-1
    plot3(c(i,1),c(i,2),c(i,3),'.','color',colors(i,:),'MarkerSize',9)
 end
 
hold off
 
 
function Err = ErrSum(w,b,x,y)
Err = 0;
for n = 1:length(x)%or length(y)
Err = Err + ( y(n) - x(n)*w-b ).^2;
end
end
