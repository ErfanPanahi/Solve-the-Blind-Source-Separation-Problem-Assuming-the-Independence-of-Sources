 %%% HW14 - BSS - Erfan Panahi 810198369
clc
clear
close all
fprintf("HW#14 - BSS - Erfan Panahi 810198369\n");

%% Definitions.
hw14 = load("hw14.mat");
S = hw14.S;
A = hw14.A;
Noise = hw14.Noise;
X = A * S + Noise;
figure
subplot(3,1,1);
plot(S.');
legend('s_1(t)','s_2(t)','s_3(t)');
title('Sources','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('S(t)','Interpreter','latex');
subplot(3,1,2);
plot((A*S).');
legend('x_1(t)','x_2(t)','x_3(t)');
title('Noiseless Observations','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('X(t)','Interpreter','latex');
subplot(3,1,3);
plot(X.');
legend('x_1(t)','x_2(t)','x_3(t)');
title('Noisy Observations','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('X(t)','Interpreter','latex');

%% Problem 1. Kurnel
warning ('off','all');
L = size(X,2);
t = 1:L;
K = @(Y) [ones(1,L);Y;Y.^2;Y.^3;Y.^4;Y.^5];
dK = @(Y) [zeros(1,L);ones(1,L);2*Y;3*Y.^2;4*Y.^3;5*Y.^4];
% init
B =[0.321,0.532,0.533; 
    0.227,0.41,0.282;
    0.321,0.821,0.81];
B = B./sqrt(sum(B.^2,2));
mu = 0.01;
Y = B * X;
ITR = 1000;
Score = zeros(1,ITR);
for itr = 1:ITR
    K1 = K(Y(1,:));
    K2 = K(Y(2,:));
    K3 = K(Y(3,:));
    Theta1 = (K1*K1.'/L)\mean(dK(Y(1,:)),2);
    Theta2 = (K2*K2.'/L)\mean(dK(Y(2,:)),2);
    Theta3 = (K3*K3.'/L)\mean(dK(Y(3,:)),2);
    Psi1 = Theta1.' * K1;
    Psi2 = Theta2.' * K2;
    Psi3 = Theta3.' * K3;
    grad_f = ([Psi1*X.' ; Psi2*X.' ; Psi3*X.'] / L) - (inv(B)).';
    B = B - mu * grad_f;
    B = B./sqrt(sum(B.^2,2));
    S_hat = B * X;
    Shat_ = S_hat;
    S_ = S;
    [~,r1] = max(abs(Shat_(1,:)*S_'));
    S_(r1,:) = 0;   
    [~,r2] = max(abs(Shat_(2,:)*S_'));
    S_(r2,:) = 0;
    [~,r3] = max(abs(Shat_(3,:)*S_'));
    S_(r3,:) = 0;
    S_hat(r1,:) = Shat_(1,:);
    S_hat(r2,:) = Shat_(2,:);
    S_hat(r3,:) = Shat_(3,:);
    S_hat(1,:) = S_hat(1,:) * max(abs(S(1,:))) / max(abs(S_hat(1,:)));
    S_hat(2,:) = - S_hat(2,:) * max(abs(S(2,:))) / max(abs(S_hat(2,:)));
    S_hat(3,:) = - S_hat(3,:) * max(abs(S(3,:))) / max(abs(S_hat(3,:))) ;
    Score(itr) = norm(grad_f)^2;
    Y = B * X;
end
figure
plot(Score);
xlim([0 100]);
title('Convergance Diagram (Kurnel)','Interpreter','latex');
xlabel('itration','Interpreter','latex');
ylabel('Score Function','Interpreter','latex');
E = norm(S - S_hat,'fro')^2/norm(S,'fro');
fprintf("Kurnel : E = %f\n",E(end));
figure
subplot(3,1,1);
plot(t,S(1,:),'b',t,S_hat(1,:),'r');
legend('s_1(t)','shat_1(t)');
title('Source 1 (Kurnel)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_1$(t)','Interpreter','latex');
subplot(3,1,2);
plot(t,S(2,:),'b',t,S_hat(2,:),'r');
legend('s_2(t)','shat_2(t)');
title('Source 2 (Kurnel)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_2$(t)','Interpreter','latex');
subplot(3,1,3);
plot(t,S(3,:),'b',t,S_hat(3,:),'r');
legend('s_3(t)','shat_3(t)');
title('Source 3 (Kurnel)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_3$(t)','Interpreter','latex');
Permutation_kurnel = B * A

%% Problem 2. Deflation
% init
B =[0.321,0.532,0.533; 
    0.227,0.41,0.282;
    0.321,0.821,0.81];
B = B./sqrt(sum(B.^2,2));
% Whitening
Rx = X * X.'; 
[U,D] = eig(Rx);
W = ((D^(-1/2))*U');
Z = W * X;
Y = B * Z;
mu = 0.01;
for itr = 1:1000
    K1 = K(Y(1,:));
    K2 = K(Y(2,:));
    K3 = K(Y(3,:));
    Theta1 = (K1*K1.'/L)\mean(dK(Y(1,:)),2);
    Theta2 = (K2*K2.'/L)\mean(dK(Y(2,:)),2);
    Theta3 = (K3*K3.'/L)\mean(dK(Y(3,:)),2);
    Psi1 = Theta1.' * K1;
    Psi2 = Theta2.' * K2;
    Psi3 = Theta3.' * K3;
    grad_H = [Psi1*Z.' ; Psi2*Z.' ; Psi3*Z.'];
    b1 = B(1,:).';
    b2 = B(2,:).';
    b3 = B(3,:).';
    b1 = b1 - mu * (grad_H(1,:)).';
    b1 = b1 / norm(b1);
    b2 = b2 - mu * (grad_H(2,:)).';
    b2 = (eye(3) - b1*b1.') * b2;
    b2 = b2 / norm(b2);
    b3 = b3 - mu * (grad_H(3,:)).';
    b3 = (eye(3) - [b1,b2]*[b1,b2].') * b3;
    b3 = b3 / norm(b3);
    B = [b1.';b2.';b3.'];
    S_hat = B * X;
    Shat_ = S_hat;
    S_ = S;
    [~,r1] = max(abs(Shat_(1,:)*S_'));
    S_(r1,:) = 0;   
    [~,r2] = max(abs(Shat_(2,:)*S_'));
    S_(r2,:) = 0;
    [~,r3] = max(abs(Shat_(3,:)*S_'));
    S_(r3,:) = 0;
    S_hat(r1,:) = Shat_(1,:);
    S_hat(r2,:) = Shat_(2,:);
    S_hat(r3,:) = Shat_(3,:);
    S_hat(1,:) = - S_hat(1,:) * max(abs(S(1,:))) / max(abs(S_hat(1,:)));
    S_hat(2,:) = S_hat(2,:) * max(abs(S(2,:))) / max(abs(S_hat(2,:)));
    S_hat(3,:) = - S_hat(3,:) * max(abs(S(3,:))) / max(abs(S_hat(3,:))) ;
    Score(itr) = norm(grad_H) / L;
    Y = B * Z;
end
figure
plot(Score);
xlim([0 100]);
title('Convergance Diagram (Deflation)','Interpreter','latex');
xlabel('itration','Interpreter','latex');
ylabel('Score Function','Interpreter','latex');
E = norm(S - S_hat,'fro')^2/norm(S,'fro');
fprintf("Deflation : E = %f\n",E(end));
figure
subplot(3,1,1);
plot(t,S(1,:),'b',t,S_hat(1,:),'r');
legend('s_1(t)','shat_1(t)');
title('Source 1 (Deflation)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_1$(t)','Interpreter','latex');
subplot(3,1,2);
plot(t,S(2,:),'b',t,S_hat(2,:),'r');
legend('s_2(t)','shat_2(t)');
title('Source 2 (Deflation)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_2$(t)','Interpreter','latex');
subplot(3,1,3);
plot(t,S(3,:),'b',t,S_hat(3,:),'r');
legend('s_3(t)','shat_3(t)');
title('Source 3 (Deflation)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_3$(t)','Interpreter','latex');
Permutation_deflation = B * W * A

%% Problem 3. Equivariant
% init
B =[0.321,0.532,0.533; 
    0.227,0.41,0.282;
    0.321,0.821,0.81];
mu = 0.05;
Y = B * X;
ITR = 1000;
E = zeros(1,ITR);
for itr = 1:ITR
    K1 = K(Y(1,:));
    K2 = K(Y(2,:));
    K3 = K(Y(3,:));
    Theta1 = (K1*K1.'/L)\mean(dK(Y(1,:)),2);
    Theta2 = (K2*K2.'/L)\mean(dK(Y(2,:)),2);
    Theta3 = (K3*K3.'/L)\mean(dK(Y(3,:)),2);
    Psi1 = Theta1.' * K1;
    Psi2 = Theta2.' * K2;
    Psi3 = Theta3.' * K3;
    grad_f = ([Psi1*X.' ; Psi2*X.' ; Psi3*X.'] / L) - (inv(B)).';
    B = (eye(3) - mu * grad_f * B.') * B;
    B = B./sqrt(sum(B.^2,2));
    S_hat = B * X;
    Shat_ = S_hat;
    S_ = S;
    [~,r1] = max(abs(Shat_(1,:)*S_'));
    S_(r1,:) = 0;   
    [~,r2] = max(abs(Shat_(2,:)*S_'));
    S_(r2,:) = 0;
    [~,r3] = max(abs(Shat_(3,:)*S_'));
    S_(r3,:) = 0;
    S_hat(1,:) = S_hat(1,:) * max(abs(S(1,:))) / max(abs(S_hat(1,:)));
    S_hat(2,:) = - S_hat(2,:) * max(abs(S(2,:))) / max(abs(S_hat(2,:)));
    S_hat(3,:) = - S_hat(3,:) * max(abs(S(3,:))) / max(abs(S_hat(3,:))) ;
    Score(itr) = norm(grad_f)^2;
    Y = B * X;
end
figure
plot(Score);
xlim([0 100]);
title('Convergance Diagram (Equivariant)','Interpreter','latex');
xlabel('itration','Interpreter','latex');
ylabel('Score Function','Interpreter','latex');
E = norm(S - S_hat,'fro')^2/norm(S,'fro');
fprintf("Equivariant : E = %f\n",E(end));
figure
subplot(3,1,1);
plot(t,S(1,:),'b',t,S_hat(1,:),'r');
legend('s_1(t)','shat_1(t)');
title('Source 1 (Equivariant)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_1$(t)','Interpreter','latex');
subplot(3,1,2);
plot(t,S(2,:),'b',t,S_hat(2,:),'r');
legend('s_2(t)','shat_2(t)');
title('Source 2 (Equivariant)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_2$(t)','Interpreter','latex');
subplot(3,1,3);
plot(t,S(3,:),'b',t,S_hat(3,:),'r');
legend('s_3(t)','shat_3(t)');
title('Source 3 (Equivariant)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_3$(t)','Interpreter','latex');
Permutation_equivarient = B * A

%% Problem 5. Kurt (deflation)
% init
B =[0.321,0.532,0.533; 
    0.227,0.41,0.282;
    0.321,0.821,0.81];
B = B./sqrt(sum(B.^2,2));
% Whitening
Rx = X * X.'; 
[U,D] = eig(Rx);
W = ((D^(-1/2))*U');
Z = W * X;
Y = B * Z;
mu = 0.01;
for itr = 1:1000
    y1 = Y(1,:);
    b1 = B(1,:).';
    kurty1 = mean(y1.^4) - 3*(mean(y1.^2))^2;
    grad_f1 = sign(kurty1)*(mean((ones(3,1)*y1.^3).*Z,2)-3*b1);
    b1 = b1 + mu * grad_f1;
    b1 = b1 / norm(b1);
    
    y2 = Y(2,:);
    b2 = B(2,:).';
    kurty2 = mean(y2.^4) - 3*(mean(y2.^2))^2;
    grad_f2 = sign(kurty2)*(mean((ones(3,1)*y2.^3).*Z,2)-3*b2);
    b2 = b2 + mu * grad_f2;
    b2 = (eye(3) - b1*b1.') * b2;
    b2 = b2 / norm(b2);
    
    y3 = Y(3,:);
    b3 = B(3,:).';
    kurty3 = mean(y3.^4) - 3*(mean(y3.^2))^2;
    grad_f3 = sign(kurty3)*(mean((ones(3,1)*y3.^3).*Z,2)-3*b3);
    b3 = b3 + mu * grad_f3;
    b3 = (eye(3) - [b1,b2]*[b1,b2].') * b3;
    b3 = b3 / norm(b3);
    
    B = [b1.';b2.';b3.'];
    S_hat = B * X;
    Shat_ = S_hat;
    S_ = S;
    [~,r1] = max(abs(Shat_(1,:)*S_'));
    S_(r1,:) = 0;   
    [~,r2] = max(abs(Shat_(2,:)*S_'));
    S_(r2,:) = 0;
    [~,r3] = max(abs(Shat_(3,:)*S_'));
    S_(r3,:) = 0;
    S_hat(r1,:) = Shat_(1,:);
    S_hat(r2,:) = Shat_(2,:);
    S_hat(r3,:) = Shat_(3,:);
    S_hat(1,:) = S_hat(1,:) * max(abs(S(1,:))) / max(abs(S_hat(1,:)));
    S_hat(2,:) = - S_hat(2,:) * max(abs(S(2,:))) / max(abs(S_hat(2,:)));
    S_hat(3,:) = - S_hat(3,:) * max(abs(S(3,:))) / max(abs(S_hat(3,:))) ;
    KURTY = mean(Y.^4,2) - 3*(mean(Y.^2,2)).^2;
    Score(itr) = norm(KURTY) / L;
    Y = B * Z;
end
figure
plot(Score);
xlim([0 100]);
title('Convergance Diagram (Kurt)','Interpreter','latex');
xlabel('itration','Interpreter','latex');
ylabel('Score Function','Interpreter','latex');
E = norm(S - S_hat,'fro')^2/norm(S,'fro');
fprintf("Kurt (deflation) : E = %f\n",E(end));
figure
subplot(3,1,1);
plot(t,S(1,:),'b',t,S_hat(1,:),'r');
legend('s_1(t)','shat_1(t)');
title('Source 1 (Kurt)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_1$(t)','Interpreter','latex');
subplot(3,1,2);
plot(t,S(2,:),'b',t,S_hat(2,:),'r');
legend('s_2(t)','shat_2(t)');
title('Source 2 (Kurt)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_2$(t)','Interpreter','latex');
subplot(3,1,3);
plot(t,S(3,:),'b',t,S_hat(3,:),'r');
legend('s_3(t) (Kurt)','shat_3(t)');
title('Source 3','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_3$(t)','Interpreter','latex');
Permutation_kurt = B * W * A

%% Problem 6. Fixed-Point
% init
B =[0.321,0.532,0.533; 
    0.227,0.41,0.282;
    0.321,0.821,0.81];
B = B./sqrt(sum(B.^2,2));
% Whitening
Rx = X * X.'; 
[U,D] = eig(Rx);
W = ((D^(-1/2))*U');
Z = W * X;
Y = B * Z;
mu = 0.01;
for itr = 1:1000
    y1 = Y(1,:);
    b1 = B(1,:).';
    b1 = mean((ones(3,1)*(y1.^3).*Z),2) - 3*b1;
    b1 = b1 / norm(b1);
    
    y2 = Y(2,:);
    b2 = B(2,:).';
    b2 = mean((ones(3,1)*(y2.^3).*Z),2) - 3*b2;
    b2 = (eye(3) - b1*b1.') * b2;
    b2 = b2 / norm(b2);
    
    y3 = Y(3,:);
    b3 = B(3,:).';
    b3 = mean((ones(3,1)*(y3.^3).*Z),2) - 3*b3;
    b3 = (eye(3) - [b1,b2]*[b1,b2].') * b3;
    b3 = b3 / norm(b3);
    
    B = [b1.';b2.';b3.'];
    S_hat = B * X;
    Shat_ = S_hat;
    S_ = S;
    [~,r1] = max(abs(Shat_(1,:)*S_'));
    S_(r1,:) = 0;   
    [~,r2] = max(abs(Shat_(2,:)*S_'));
    S_(r2,:) = 0;
    [~,r3] = max(abs(Shat_(3,:)*S_'));
    S_(r3,:) = 0;
    S_hat(r1,:) = Shat_(1,:);
    S_hat(r2,:) = Shat_(2,:);
    S_hat(r3,:) = Shat_(3,:);
    S_hat(1,:) = S_hat(1,:) * max(abs(S(1,:))) / max(abs(S_hat(1,:)));
    S_hat(2,:) = - S_hat(2,:) * max(abs(S(2,:))) / max(abs(S_hat(2,:)));
    S_hat(3,:) = - S_hat(3,:) * max(abs(S(3,:))) / max(abs(S_hat(3,:))) ;
    KURTY = mean(Y.^4,2) - 3*(mean(Y.^2,2)).^2;
    Score(itr) = norm(KURTY) / L;
    Y = B * Z;
end
figure
plot(Score);
xlim([0 100]);
title('Convergance Diagram (Fixed-Point)','Interpreter','latex');
xlabel('itration','Interpreter','latex');
ylabel('Score Function','Interpreter','latex');
fprintf("Fixed-Point : E = %f\n",E(end));
figure
subplot(3,1,1);
plot(t,S(1,:),'b',t,S_hat(1,:),'r');
legend('s_1(t)','shat_1(t)');
title('Source 1 (Fixed-Point)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_1$(t)','Interpreter','latex');
subplot(3,1,2);
plot(t,S(2,:),'b',t,S_hat(2,:),'r');
legend('s_2(t)','shat_2(t)');
title('Source 2 (Fixed-Point)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_2$(t)','Interpreter','latex');
subplot(3,1,3);
plot(t,S(3,:),'b',t,S_hat(3,:),'r');
legend('s_3(t) (Fixed-Point)','shat_3(t)');
title('Source 3','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_3$(t)','Interpreter','latex');
Permutation_Fixed_Point = B * W * A

%% Problem 7. G.p
% init
B =[0.321,0.532,0.533; 
    0.227,0.41,0.282;
    0.321,0.821,0.81];
B = B./sqrt(sum(B.^2,2));
v = randn(1,L);
% Whitening
Rx = X * X.'; 
[U,D] = eig(Rx);
W = ((D^(-1/2))*U');
Z = W * X;
Y = B * Z;
mu = 0.01;
for itr = 1:1000
    Gv = -exp(-v.^2/2);
    y1 = Y(1,:);
    b1 = B(1,:).';
    Gy1 = -exp(-y1.^2/2);
    gy1 = -y1 .* Gy1;
    fb1 = (mean(Gy1) - mean(Gv))^2;
    grad_f1 = fb1*(mean((ones(3,1)*gy1).*Z,2));
    b1 = b1 + mu * grad_f1;
    b1 = b1 / norm(b1);
    
    y2 = Y(2,:);
    b2 = B(2,:).';
    Gy2 = -exp(-y2.^2/2);
    gy2 = -y2 .* Gy2;
    fb2 = (mean(Gy2) - mean(Gv))^2;
    grad_f2 = fb2*(mean((ones(3,1)*gy2).*Z,2));
    b2 = b2 + mu * grad_f2;
    b2 = (eye(3) - b1*b1.') * b2;
    b2 = b2 / norm(b2);
    
    y3 = Y(3,:);
    b3 = B(3,:).';
    Gy3 = -exp(-y3.^2/2);
    gy3 = -y3 .* Gy3;
    fb3 = (mean(Gy3) - mean(Gv))^2;
    grad_f3 = fb3*(mean((ones(3,1)*gy3).*Z,2));
    b3 = b3 + mu * grad_f3;
    b3 = (eye(3) - [b1,b2]*[b1,b2].') * b3;
    b3 = b3 / norm(b3);
    
    B = [b1.';b2.';b3.'];
    S_hat = B * X;
    Shat_ = S_hat;
    S_ = S;
    [~,r1] = max(abs(Shat_(1,:)*S_'));
    S_(r1,:) = 0;   
    [~,r2] = max(abs(Shat_(2,:)*S_'));
    S_(r2,:) = 0;
    [~,r3] = max(abs(Shat_(3,:)*S_'));
    S_(r3,:) = 0;
    S_hat(r1,:) = Shat_(1,:);
    S_hat(r2,:) = Shat_(2,:);
    S_hat(r3,:) = Shat_(3,:);
    S_hat(1,:) = S_hat(1,:) * max(abs(S(1,:))) / max(abs(S_hat(1,:)));
    S_hat(2,:) = - S_hat(2,:) * max(abs(S(2,:))) / max(abs(S_hat(2,:)));
    S_hat(3,:) = - S_hat(3,:) * max(abs(S(3,:))) / max(abs(S_hat(3,:))) ;
    KURTY = mean(Y.^4,2) - 3*(mean(Y.^2,2)).^2;
    Score(itr) = norm(KURTY) / L;
    Y = B * Z;
end
figure
plot(Score);
xlim([0 100]);
title('Convergance Diagram (G.P)','Interpreter','latex');
xlabel('itration','Interpreter','latex');
ylabel('Score Function','Interpreter','latex');
E = norm(S - S_hat,'fro')^2/norm(S,'fro');
fprintf("G.P : E = %f\n",E(end));
figure
subplot(3,1,1);
plot(t,S(1,:),'b',t,S_hat(1,:),'r');
legend('s_1(t)','shat_1(t)');
title('Source 1 (G.P)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_1$(t)','Interpreter','latex');
subplot(3,1,2);
plot(t,S(2,:),'b',t,S_hat(2,:),'r');
legend('s_2(t)','shat_2(t)');
title('Source 2 (G.P)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_2$(t)','Interpreter','latex');
subplot(3,1,3);
plot(t,S(3,:),'b',t,S_hat(3,:),'r');
legend('s_3(t) (G.P)','shat_3(t)');
title('Source 3','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_3$(t)','Interpreter','latex');
Permutation_GP = B * W * A

%% Problem 8. Fast ICA
% init
B =[0.321,0.532,0.533; 
    0.227,0.41,0.282;
    0.321,0.821,0.81];
B = B./sqrt(sum(B.^2,2));
Y = B * X;
mu = 0.01;
for itr = 1:1000
    Gv = -exp(-v.^2/2);
    y1 = Y(1,:);
    b1 = B(1,:).';
    Gy1 = -exp(-y1.^2/2);
    gy1 = -y1 .* Gy1;
    ggy1 = (y1.^2 -1) .* Gy1;
    b1 = mean((ones(3,1)*gy1).*X,2) - mean(ggy1)*b1;
    b1 = b1 / norm(b1);
    
    y2 = Y(2,:);
    b2 = B(2,:).';
    Gy2 = -exp(-y2.^2/2);
    gy2 = -y2 .* Gy2;
    ggy2 = (y2.^2 -1) .* Gy2;
    b2 = mean((ones(3,1)*gy2).*X,2) - mean(ggy2)*b2;
    b2 = b2 / norm(b2);
    b2 = (eye(3) - b1*b1.') * b2;
    b2 = b2 / norm(b2);
    
    y3 = Y(3,:);
    b3 = B(3,:).';
    Gy3 = -exp(-y3.^2/2);
    gy3 = -y3 .* Gy3;
    ggy3 = (y3.^2 -1) .* Gy3;
    b3 = mean((ones(3,1)*gy3).*X,2) - mean(ggy3)*b3;
    b3 = b3 / norm(b3);
    b3 = (eye(3) - [b1,b2]*[b1,b2].') * b3;
    b3 = b3 / norm(b3);
    
    B = [b1.';b2.';b3.'];
    S_hat = B * X;
    Shat_ = S_hat;
    S_ = S;
    [~,r1] = max(abs(Shat_(1,:)*S_'));
    S_(r1,:) = 0;   
    [~,r2] = max(abs(Shat_(2,:)*S_'));
    S_(r2,:) = 0;
    [~,r3] = max(abs(Shat_(3,:)*S_'));
    S_(r3,:) = 0;
    S_hat(r1,:) = Shat_(1,:);
    S_hat(r2,:) = Shat_(2,:);
    S_hat(r3,:) = Shat_(3,:);
    S_hat(1,:) = S_hat(1,:) * max(abs(S(1,:))) / max(abs(S_hat(1,:)));
    S_hat(2,:) = - S_hat(2,:) * max(abs(S(2,:))) / max(abs(S_hat(2,:)));
    S_hat(3,:) = - S_hat(3,:) * max(abs(S(3,:))) / max(abs(S_hat(3,:))) ;
    KURTY = mean(Y.^4,2) - 3*(mean(Y.^2,2)).^2;
    Score(itr) = norm(KURTY) / L;
    Y = B * Z;
end
figure
plot(Score);
xlim([0 100]);
title('Convergance Diagram (FAST ICA)','Interpreter','latex');
xlabel('itration','Interpreter','latex');
ylabel('Score Function','Interpreter','latex');
E = norm(S - S_hat,'fro')^2/norm(S,'fro');
fprintf("FAST ICA : E = %f\n",E(end));
figure
subplot(3,1,1);
plot(t,S(1,:),'b',t,S_hat(1,:),'r');
legend('s_1(t)','shat_1(t)');
title('Source 1 (FAST ICA)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_1$(t)','Interpreter','latex');
subplot(3,1,2);
plot(t,S(2,:),'b',t,S_hat(2,:),'r');
legend('s_2(t)','shat_2(t)');
title('Source 2 (FAST ICA)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_2$(t)','Interpreter','latex');
subplot(3,1,3);
plot(t,S(3,:),'b',t,S_hat(3,:),'r');
legend('s_3(t) (G.P)','shat_3(t)');
title('Source 3 (FAST ICA)','Interpreter','latex');
xlabel('time','Interpreter','latex');
ylabel('$S_3$(t)','Interpreter','latex');
Permutation_FAST_ICA = B * A