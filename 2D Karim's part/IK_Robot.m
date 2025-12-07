%%% IK Robot
clc; clear; close all;

% Robot parameters
L_thigh = 0.35;  
L_shank = 0.35;  
L_foot  = 0.1;  
hip_height = 0.6;

% Walking parameters
step_length = 0.3;
step_height = 0.12;   
num_steps = 4;
frames_per_step = 100;
total_frames = num_steps * frames_per_step;
hip_osc = 0.02;      

ground_level = 0;  % ensure no penetration

%% Foot swing trajectory (smooth, controls only height)
foot_swing = @(t) [-step_length/2 + step_length*t, ...
                   step_height*sin(pi*t)];

%% Inverse Kinematics (3 DOF), with SAFE acos and FLAT FOOT
clamp = @(v) min(max(v,-1),1); 

ik_leg = @(x,z) deal( ...
    atan2(-z,x) - atan2( L_shank.*sqrt(1 - clamp((x.^2+z.^2-L_thigh^2-L_shank^2) ...
                   /(2*L_thigh*L_shank)).^2), ...
                   L_thigh + L_shank.*clamp((x.^2+z.^2-L_thigh^2-L_shank^2) ...
                   /(2*L_thigh*L_shank)) ), ...
    acos( clamp((x.^2+z.^2-L_thigh^2-L_shank^2)/(2*L_thigh*L_shank)) ), ...
    0 );  % ankle will be set flat later

%% Forward Kinematics
fk_leg = @(hip,knee,ankle,hip_pos) [ ...
    hip_pos; ...
    hip_pos + [L_thigh*cos(hip), -L_thigh*sin(hip)]; ...
    hip_pos + [L_thigh*cos(hip)+L_shank*cos(hip+knee), ...
               -L_thigh*sin(hip)-L_shank*sin(hip+knee)]; ...
    hip_pos + [L_thigh*cos(hip)+L_shank*cos(hip+knee)+L_foot*cos(hip+knee+ankle), ...
               -L_thigh*sin(hip)-L_shank*sin(hip+knee)-L_foot*sin(hip+knee+ankle)] ...
];

%% Initialize
stance_foot = [0, ground_level];
hip_pos     = [0, hip_height];
left_swing  = true;

%% Figure setup
figure('Color','w'); axis equal; axis([-0.5 2 -0.2 0.8]); grid on; hold on;

h_left  = plot([0 0],[0 0],'o-','LineWidth',3,'Color','b');
h_right = plot([0 0],[0 0],'o-','LineWidth',3,'Color','r');

% hip
h_torso = plot(hip_pos(1), hip_pos(2)+0.25, 'ko', 'MarkerSize', 12, 'MarkerFaceColor','k');

h_hip = plot(hip_pos(1),hip_pos(2),'ko','MarkerSize',8,'MarkerFaceColor','k');

%% Animation loop
for f = 1:total_frames
    t = mod(f-1, frames_per_step)/frames_per_step;

    % Swing foot trajectory
    swing_rel = foot_swing(t);
    swing_pos = stance_foot + swing_rel;

    % Hip motion
    hip_x = stance_foot(1) + 0.5*(swing_pos(1)-stance_foot(1));
    hip_z = hip_height + hip_osc*sin(pi*t);
    hip_pos = [hip_x, hip_z];

    %% Inverse kinematics for both legs
    if left_swing
        % Left leg is swing
        [hip_l, knee_l, ankle_l] = ik_leg(swing_pos(1)-hip_pos(1), swing_pos(2)-hip_pos(2));
        ankle_l = -(hip_l + knee_l); % enforce flat foot
                
        % Right leg is stance
        [hip_r, knee_r, ~] = ik_leg(stance_foot(1)-hip_pos(1), stance_foot(2)-hip_pos(2));
        ankle_r = -(hip_r + knee_r);

    else
        % Right leg is swing
        [hip_r, knee_r, ankle_r] = ik_leg(swing_pos(1)-hip_pos(1), swing_pos(2)-hip_pos(2));
        ankle_r = -(hip_r + knee_r);

        % Left leg is stance
        [hip_l, knee_l, ~] = ik_leg(stance_foot(1)-hip_pos(1), stance_foot(2)-hip_pos(2));
        ankle_l = -(hip_l + knee_l);
    end

    %% Forward kinematics
    pts_left  = fk_leg(hip_l,knee_l,ankle_l,hip_pos);
    pts_right = fk_leg(hip_r,knee_r,ankle_r,hip_pos);

    %% Update plots
    set(h_left,'XData',pts_left(:,1),'YData',pts_left(:,2));
    set(h_right,'XData',pts_right(:,1),'YData',pts_right(:,2));
    set(h_hip,'XData',hip_pos(1),'YData',hip_pos(2));
    set(h_torso,'XData',hip_pos(1),'YData',hip_pos(2)+0.25);

    drawnow;

    %% Switch stance leg
    if mod(f, frames_per_step) == 0
        stance_foot = swing_pos;
        stance_foot(2) = ground_level;  % enforce no sinking
        left_swing = ~left_swing;
    end
end
