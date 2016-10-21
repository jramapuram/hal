% Initialization

D1 = ones(1,10);
D1 = diag(exp(i*D1));

D2 = ones(1,10);
D2 = diag(exp(i*D2));

D3 = ones(1,10);
D3 = diag(exp(i*D3));

R1 = ones(1,10)+i;
R1 = R1/norm(R1);
R1 = eye(10)-2*R1'*R1;

R2 =  ones(1,10)+i;
R2 = R2/norm(R2);
R2 = eye(10)-2*R2'*R2;

V = ones(10)+i;

U = ones(20,10);

Pi = eye(10);
Pi = Pi([1 2 3 4 5 6 7 8 9 10],:);

b1 = ones(1,10);
b2 = ones(1,10);

inputs = [0.4 -1.2 -0.55 0.15 -0.55 3.2 -2.5 3.2 -12 30];

h0 = ones(1,10)/2;

%-------------------------------------------------------------
%z = Wh0 + Vx
z = ifft(fft(h*D1,[],2)*R1*Pi*D2,[],2)*R2*D3 + inputs*V;

%h1 = modRelu(z, b1)
newMod = abs(z)+b1;
newMod(newMod<0) = 0;
h1 = z.*newMod./abs(z);


h1_concat = [real(h1) imag(h1)];

%o = U*h1 + b2
out = h1_concat*U;
out = out + b2;


