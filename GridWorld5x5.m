%% E. Öner Tartan
% 06.01.2023
% onertartan@gmail.com
% 5x5 GridWorld example(Exercise 3.5 from R. S. Sutton ve A. G. Barto, 
% Introduction to Reinforcement Learning, Cambridge, MA., MIT Press/Bradford Books, 2018, p. 60.)
% which calculates state value function V by value iteration 

% INITIALIZATION
clc,clear
% A: a Map object (dictionary) mapping action space to corresponding transition effects
A = containers.Map(["north","south","east","west"],{[-1,0],[1,0],[0,1],[0,-1]});
% V: State values matrix 
V = zeros(5,5);
%Theta: Threshold value used for termination
theta=.00001;
% delta: max difference of values between consecutive sweeps throught state space
delta = theta;
gamma = 0.9;

while delta>=theta  
   delta = 0; 
   
    for i=1:5
        for j=1:5          
            v_new = bellman_update(i,j,V,A,gamma);
            delta = max(delta,abs(v_new-V(i,j) ));   %update delta
            V(i,j)=v_new;                            % update V(i,j) in place
        end
    end    
   
end
V

function v_new=bellman_update(i,j,V, A, gamma)
    v_new=0;
    %vnew is calculated according to Bellman equation for state value function
    for action=keys(A)
      a= action{1};
      [i_new,j_new,reward]=action_and_transition(i,j,a,A);      %state transition due to the action taken
      v_new= v_new+0.25*( gamma *V(i_new,j_new)+reward);        % bellman_update with p(s,a)=0.25 --> equiprobable actions
    end
end
function [i_new,j_new,reward]= action_and_transition(i,j,a,A)
    % returns new state's coordinates and the reward
    % based on the current state's coordinates and the action taken
    reward=0;
    s_new= [i,j]+A(a);
    i_new=s_new(1);
    j_new=s_new(2);
 
    if (i==1 && j==2)                                %from (1,2) agent jumps to (5,2) and receives +10 reward
            reward=10; 
            s_new= [5,2];
    elseif (i==1 && j==4)                            %from (1,4) agent jumps to (3,4) and receives +5 reward
            reward=5; 
             s_new= [3,4];
    elseif i_new<1 || j_new<1 || i_new>5 || j_new>5  %actions towards out of the grid: agent stays in the same state and receives -1 reward
        s_new= [i,j];
        reward=-1;
    end
i_new=s_new(1);
j_new=s_new(2);
end
