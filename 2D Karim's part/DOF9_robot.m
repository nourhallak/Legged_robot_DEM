%% 9DOF Robot
%% Symbolic variables (generalized coordinates & derivatives)
clear; clc; close all;

syms xh yh phih thL phL psL thR phR psR real
syms d_xh d_yh d_phih d_thL d_phL d_psL d_thR d_phR d_psR real
q  = [xh; yh; phih; thL; phL; psL; thR; phR; psR];
dq = [d_xh; d_yh; d_phih; d_thL; d_phL; d_psL; d_thR; d_phR; d_psR];

syms dd_xh dd_yh dd_phih dd_thL dd_phL dd_psL dd_thR dd_phR dd_psR real
ddq = [dd_xh; dd_yh; dd_phih; dd_thL; dd_phL; dd_psL; dd_thR; dd_phR; dd_psR];

% Robot parameters
syms L_th L_sh L_ft m_th m_sh m_ft m_hip I_th I_sh I_ft I_hip g real

%% Positions
% Hip/pelvis position
P_hip = [xh; yh];

% Left leg
P_kneeL  = P_hip - [ L_th*cos(thL); L_th*sin(thL) ];
P_ankleL = P_kneeL - [ L_sh*cos(phL); L_sh*sin(phL) ];
P_footL  = P_ankleL - [ L_ft*cos(psL); L_ft*sin(psL) ];

% Right leg
P_kneeR  = P_hip - [ L_th*cos(thR); L_th*sin(thR) ];
P_ankleR = P_kneeR - [ L_sh*cos(phR); L_sh*sin(phR) ];
P_footR  = P_ankleR - [ L_ft*cos(psR); L_ft*sin(psR) ];


%% Velocities via Jacobians
v_hip     = jacobian(P_hip,q)*dq;
v_kneeL   = jacobian(P_kneeL,q)*dq;
v_ankleL  = jacobian(P_ankleL,q)*dq;
v_footL   = jacobian(P_footL,q)*dq;

v_kneeR   = jacobian(P_kneeR,q)*dq;
v_ankleR  = jacobian(P_ankleR,q)*dq;
v_footR   = jacobian(P_footR,q)*dq;

%% Kinetic energy
T = 0.5*m_hip*(v_hip.'*v_hip) + 0.5*I_hip*d_phih^2 + ...
    0.5*m_th*(v_kneeL.'*v_kneeL) + 0.5*I_th*d_thL^2 + ...
    0.5*m_sh*(v_ankleL.'*v_ankleL) + 0.5*I_sh*d_phL^2 + ...
    0.5*m_ft*(v_footL.'*v_footL) + 0.5*I_ft*d_psL^2 + ...
    0.5*m_th*(v_kneeR.'*v_kneeR) + 0.5*I_th*d_thR^2 + ...
    0.5*m_sh*(v_ankleR.'*v_ankleR) + 0.5*I_sh*d_phR^2 + ...
    0.5*m_ft*(v_footR.'*v_footR) + 0.5*I_ft*d_psR^2;

%% Potential energy
V = g*( m_hip*P_hip(2) + ...
        m_th*P_kneeL(2) + m_sh*P_ankleL(2) + m_ft*P_footL(2) + ...
        m_th*P_kneeR(2) + m_sh*P_ankleR(2) + m_ft*P_footR(2) );

%% Lagrange equations
Lagr = T - V;
n = length(q);
EOM = sym(zeros(n,1));
for i = 1:n
    dL_dqdot = diff(Lagr, dq(i));
    ddt_term = jacobian(dL_dqdot,q)*dq + jacobian(dL_dqdot,dq)*ddq;
    EOM(i) = simplify(ddt_term - diff(Lagr, q(i)));
end

%% Mass matrix and forcing vector
[M,f] = equationsToMatrix(EOM, ddq);
M = simplify(M);
f = simplify(f);

disp('Mass matrix M(q):'); disp(M);
disp('RHS forcing vector f(q,dq):'); disp(f);
