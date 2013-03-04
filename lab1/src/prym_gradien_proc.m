function [A, p, usl, nusl] = prym_gradien_proc()

global s q ;
s = 2;              !число регрессоров
q = s*(s+1)/2+1;    !число точек

A = [[1; 1] [-1; 2] [1; -2] [3; 0]];   
for i = 1:1:q
    p(i) = 1/q;
end;

usl = 1.e+11;

detM = det(infM(A,p));

maxSpM = trace((infM(A,p)^-1)*(my_f(A(:,1))*my_f(A(:,1))'));
for i = 1:2:q
    if (maxSpM < trace((infM(A,p)^-1)*(my_f(A(:,i))*my_f(A(:,i))')))
        maxSpM = trace((infM(A,p)^-1)*(my_f(A(:,i))*my_f(A(:,i))'));
    end
end
save('E:\\outfile.txt','-ascii','A','p','detM','maxSpM','-double');

!save('D:\\nstu\\10sem\\ћћѕЁ\\laba1\\outfile.txt','-ascii','s');
while (usl > 1.e-3)    
    !критерий D-оптимальности
    options = optimset('Algorithm','interior-point','GradObj', 'off', 'Display', 'iter', 'MaxIter', 20);
    lb = zeros(1,q); 
    ub(1:q) = 1;
    aeq(1:q) = 1; 

    [p1,fval]=fmincon(@(p_)func_p_D(A,p_),p,[],[],aeq,1,lb,ub,[],options);

    lb = zeros(s,q);
    lb(:,:) = -5; 
    
    ub = zeros(s,q);
    ub(:,:) = 5;
    
    options = optimset('Algorithm','interior-point','GradObj', 'off', 'Display', 'iter', 'MaxIter', 20);
    [A1,fval]=fmincon(@(A_)func_alfa_D(A_,p),A,[],[],[],[],lb,ub,[],options);


    usl = 0;
    for i = 1:1:q
        usl = usl + norm(A1(:,i)-A(:,i))+(p1(i)-p(i))^2;
    end

    A=A1;
    p=p1;
    detM = det(infM(A,p));
        
    maxSpM = trace((infM(A,p)^-1)*(my_f(A(:,1))*my_f(A(:,1))'));
    for i = 1:2:q
        if (maxSpM < trace((infM(A,p)^-1)*(my_f(A(:,i))*my_f(A(:,i))')))
            maxSpM = trace((infM(A,p)^-1)*(my_f(A(:,i))*my_f(A(:,i))'));
        end
    end
    save('E:\\outfile.txt','-ascii','A','p','detM','maxSpM','-double','-append');

end

nusl = zeros(1,q);
for i = 1:1:q
    nusl(i) = abs(trace((infM(A,p)^-1)*(my_f(A(:,i))*my_f(A(:,i))'))-s);
end

function [f] = func_alfa_D(A,p)
! вычислние градиента по A дл€ D оптималносьти
f = -log(det(infM(A,p)));


function [f, g] = func_p_D(A,p) 
! вычислние градиента по р дл€ D оптималносьти   
global q ;
M = infM(A,p);
g = zeros(1,q);
for i = 1:1:q
    g(i) = -trace(M^-1*(my_f(A(:,i))*my_f(A(:,i))'));
end;
f = -log(det(infM(A,p)));


function [M] = infM(A,p)
global s q ;
M = zeros(s,s);
for i = 1:1:q
    M = M + my_f(A(:,i))*my_f(A(:,i))'*p(i);
end;

function [f] = my_f(Alfa)
x = Alfa(1);
y = Alfa(2)*Alfa(2);
f = [x; y];

