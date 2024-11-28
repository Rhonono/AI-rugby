import pygame
import math
import numpy as np
import random
import sys
import json
import os
import time

# Configuration de la fenêtre
WIDTH, HEIGHT = 1500, 1000
FPS = 60
selected_player_input = None
# Couleurs
RED_B = (255, 0, 0)
RED = (255, 0, 0)
RED_L = (255, 100, 0)
RED_C = (255, 0, 100)
BLUE_B = (0, 0, 255)
BLUE = (0, 0, 255)
BLUE_L = (0, 100, 255)
BLUE_C = (100, 0, 255)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)  # Ballon
selected_player = None

class Goal:
    def __init__(self, x, y, team):
        self.x = x
        self.y = y
        self.team = team  # 0 pour bleu, 1 pour rouge  

# Classe Ball
class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.holder = None  # Aucun joueur ne possède la balle au départ
        self.speed = 0 
        self.target_x = None
        self.target_Y = None


    def update_position(self):
        if self.holder:  # Si la balle est contrôlée, elle suit son propriétaire
            self.x = self.holder.x
            self.y = self.holder.y
        else:
         if self.speed is not None and self.target_x is not None and self.target_y is not None:
                # Calculer l'orientation vers la cible
                balle_orientation = math.atan2(self.target_y - self.y, self.target_x - self.x)

                # Vérifier si la balle a atteint ou dépassé le point cible
                dx = self.target_x - self.x
                dy = self.target_y - self.y

                # Si la distance est inférieure à la vitesse, considérer la cible atteinte
                if math.hypot(dx, dy) <= self.speed:
                    self.x = self.target_x
                    self.y = self.target_y
                    self.speed = None
                    self.target_x = None
                    self.target_y = None
                else:
                    # Déplacer la balle dans la direction de la cible
                    self.x += math.cos(balle_orientation) * self.speed
                    self.y += math.sin(balle_orientation) * self.speed            


    def draw(self, screen):
        pygame.draw.circle(screen, ORANGE, (int(self.x), int(self.y)), 8)

# Classe Player
class Player:    
    def __init__(self, x, y, orientation, team, speed=2, force=2, nn=None, generation=None,  profile=None):
        self.x = x
        self.y = y
        self.orientation = orientation  # Orientation en radians
        self.team = team  # 0 pour rouge, 1 pour bleu
        self.cooldown_timer = 0  # Moment où le joueur a perdu la balle
        self.cooldown_duration = 2  # Durée en secondes pendant laquelle il ne peut pas ramasser la balle
        self.has_ball = False
        self.nn = nn if nn is not None else NeuralNetwork(13, 26 ,20, 3)  # Réseau neuronal avec 8 entrées, 10 neurones cachés, 2 sorties
        self.speed = 2
        self.rotation_speed = 0.5
        self.reward_total = 0  # Récompense totale accumulée
        self.intial_x = x
        self.intial_y = y
        self.last_boost_time = -3  # Initialiser à -3 pour permettre un boost dès le début
        self.boost_duration = 0.5  # Durée du boost en secondes
        self.boosting = False
        self.boost_end_time = 0
        self.force=2
        self.generation = generation if generation is not None else 1
        self.immobile_time = 0  # Temps d'immobilité accumulé
        self.previous_position = (x, y)  # Position précédente
        # Assigner un profil au joueur
        self.profile = profile if profile is not None else random.choice(["bloqueur", "linemen", "courreur"]) 
        
        

        # Définir les caractéristiques en fonction du profil
        if profile == "bloqueur":
            self.force = 7
            self.speed = 2
            self.color = BLUE_B if self.team == 1 else RED_B
            self.boost_duration = 0.35
        elif profile == "linemen":
            self.force = 3.5
            self.speed = 4
            self.color = BLUE_L if self.team == 1 else RED_L
        elif profile == "courreur":
            self.force = 1.5
            self.speed = 6
            self.color = BLUE_C if self.team == 1 else RED_C
            self.boost_duration = 0.7
        else:  
            profile = "linemen"
            self.force = 2
            self.speed = 2
            self.color = BLUE_L if self.team == 1 else RED_L   
    
 
    
    def get_opponents_above_ratio(self, opponents):
        """
        Calcule le ratio des adversaires ayant une position Y supérieure à celle du joueur.

        :param opponents: Liste des joueurs adverses.
        :return: Ratio (float).
        """
        if not opponents:
            return 0.0  # Aucun adversaire, ratio = 0

        above_count = sum(1 for opponent in opponents if opponent.y > self.y)
        return above_count / len(opponents)

    def change_color(self):
        if self.profile == "bloqueur":
            self.force = 7
            self.speed = 2
            self.color = BLUE_B if self.team == 1 else RED_B
            self.boost_duration = 0.35
        elif self.profile == "linemen":
            self.force = 3.5
            self.speed = 4
            self.color = BLUE_L if self.team == 1 else RED_L
        elif self.profile == "courreur":
            self.force = 1.5
            self.speed = 6
            self.color = BLUE_C if self.team == 1 else RED_C
            self.boost_duration = 0.7
        else:  
            self.profile = "linemen"
            self.force = 2
            self.speed = 2
            self.color = BLUE_L if self.team == 1 else RED_L    
            return  

    def check_immobility(self, ball, dt):
         """
         Vérifie si le joueur est immobile et lâche la balle si nécessaire.
         
              :param ball: L'objet balle.
             :param enbuts: Dictionnaire des en-buts {"red": (x, y), "blue": (x, y)}.
             :param dt: Temps écoulé depuis la dernière mise à jour.
              """
         # Vérifiez si la position actuelle est identique à la précédente
         if (self.x, self.y) == self.previous_position:
             self.immobile_time += dt
         else:
             self.immobile_time = 0  # Réinitialiser le temps d'immobilité si le joueur a bougé

         # Si le temps d'immobilité dépasse 2,5 secondes, lâcher la balle
         if self.immobile_time >= 2.5 and self.has_ball:
          self.drop_ball(ball)
          target_enbut = (self.x+50,self.y) if self.team == 1 else (self.x-50,self.y)
          ball.target_x, ball.target_y = target_enbut
          ball.speed = 6  # Une vitesse définie pour la balle

         # Mettre à jour la position précédente
         self.previous_position = (self.x, self.y)

    def activate_boost(self):
       """Active un boost temporaire de vitesse si le cooldown est écoulé."""
       current_time = time.time()
       if current_time - self.last_boost_time >= 3:  # Cooldown de 3 secondes
            self.boosting = True
            self.last_boost_time = current_time
            self.boost_end_time = current_time + self.boost_duration
            self.speed += 2  # Doubler la vitesse
            #print(f"Boost activé pour le joueur {self} à {current_time:.2f} secondes.")

 
    def check_score_and_reset(self, ball, WIDTH, HEIGHT, score, all_players):
        """
        Vérifie si le joueur a marqué et réinitialise les positions de tous les joueurs et de la balle.
        """
        if not self.has_ball:
          return  # Pas de balle, pas de marquage possible
        
        # pas de balle en touche
        if self.y < 50:
            self.drop_ball(ball, 200)
            return
        
        if self.y > HEIGHT-50:
            self.drop_ball(ball, -200)
            return

            

        # Vérification des conditions de marquage
        if (self.team == 1 and self.x <= 50) or (self.team == 0 and self.x >= WIDTH - 50):
        # Ajouter des points à l'équipe
          if self.team == 1:  # Équipe bleue
              score['blue'] += 3
          else:  # Équipe rouge
              score['red'] += 3

          # Récompense pour le joueur qui marque
          self.reward_total += 5000000

           # Réinitialisation des positions de tous les joueurs
          for player in all_players:
              player.x = random.randint(50, WIDTH / 2 - 50) if player.team == 0 else random.randint(WIDTH / 2 + 50, WIDTH -50)  # Position aléatoire
              player.y = random.randint(50, HEIGHT - 50)
        
          self.has_ball = False  # Aucun joueur n'a la balle au départ

         # Réinitialisation de la balle au centre
          ball.x = WIDTH // 2
          ball.y = HEIGHT // 2
          ball.holder = None
        
          
        if (self.team == 0 and self.x <= 40) or (self.team == 1 and self.x >= WIDTH - 40):
            # Réinitialisation de la balle au centre
            ball.x = WIDTH // 2
            ball.y = HEIGHT // 2
            ball.holder = None
            self.has_ball=False

  
    def constrain_position(self):
        MARGIN = 10  # Optionnel : marges pour éviter les bords
        self.x = max(MARGIN, min(self.x, WIDTH - MARGIN))
        self.y = max(MARGIN, min(self.y, HEIGHT - MARGIN))

    def angle_between_orientation_and_segment(self, target):
        dx = target.x - self.x
        dy = target.y - self.y
        target_angle = math.atan2(dy, dx)
        angle_difference = (target_angle - self.orientation) % (2 * math.pi)
        if angle_difference > math.pi:
            angle_difference -= 2 * math.pi
        return angle_difference

    def get_nearest_player(self, all_players):
        return min(
            (player for player in all_players if player is not self),
            key=lambda p: math.hypot(self.x - p.x, self.y - p.y),
            default=None
        )

    def pick_up_ball(self, ball):
        current_time = time.time()
        if current_time - self.cooldown_timer < self.cooldown_duration:
         return  # Ne pas permettre de ramasser la balle
        if math.hypot(self.x - ball.x, self.y - ball.y) < 40:  # Ramassage si proche de la balle
            if ball.holder != None:
                if ball.holder.team != self.team:
                   ball.holder.drop_ball(ball)
                else:
                   return    
            ball.holder = self
            self.has_ball = True
            self.show_pickup_text(math.hypot(self.x - ball.x, self.y - ball.y))
    
    def show_pickup_text(self, info):
        """Affiche le texte 'Pickup' près du joueur."""
        font = pygame.font.Font(None, 36)
        text = font.render("Pickup " + str(info) , True, (255, 255, 255))  # Texte blanc
        screen.blit(text, (self.x - 25, self.y - 20))  # Affiche à une position relative au joueur

    def drop_ball(self, ball, xdecal=0):
        if self.has_ball:
            ball.holder = None
            self.has_ball = False
            ball.y=ball.y + xdecal
            self.cooldown_timer = time.time()
            self.reward_total -= 100

    def get_nearest_enemy_in_triangle(self, opponents):
       """
       Trouve la distance vers l'ennemi le plus proche dans un triangle isocèle
       dont la hauteur est perpendiculaire à la ligne d'en-but adverse, 
       sans tenir compte de l'orientation du joueur.

       :param opponents: Liste des joueurs adverses.
       :return: Distance vers l'ennemi le plus proche ou float('inf') si aucun ennemi n'est trouvé.
       """
       angle_half = math.pi / 30  # Demi-angle d'ouverture (π/6 total)

            # Direction perpendiculaire à la ligne d'en-but
       if self.team == 0:  # L'équipe rouge vise l'en-but droit
            direction_to_goal = 0  # Axe horizontal vers la droite
       else:  # L'équipe bleue vise l'en-but gauche
            direction_to_goal = math.pi  # Axe horizontal vers la gauche
    
       nearest_distance = float('inf')
    
       for opponent in opponents:
        # Calcul de l'angle vers l'opposant sans tenir compte de l'orientation du joueur
         angle_to_opponent = math.atan2(opponent.y - self.y, opponent.x - self.x)
         distance_to_opponent = math.hypot(self.x - opponent.x, self.y - opponent.y)

        # Vérifier si l'opposant est dans le triangle (relatif à la direction perpendiculaire à l'en-but)
         if abs(angle_to_opponent - direction_to_goal) <= angle_half:
             nearest_distance = min(nearest_distance, distance_to_opponent)
    
       return nearest_distance

    def decide(self, ball):
        global selected_player, selected_player_input
        opponents = blue_team if self.team == 1 else red_team
        if self.has_ball==True and ((self.team == 0 and self.x <= 40) or (self.team == 1 and self.x >= WIDTH - 40)):
            # Réinitialisation de la balle au centre
            factor= 1 if self.team ==0 else -1
            ball.x -= factor*100
            ball.holder = None
            self.has_ball=False
        
        # Angle et distance vers la balle
        
        
        angle_ball = self.angle_between_orientation_and_segment(ball)
        distance_ball = math.hypot(self.x - ball.x, self.y - ball.y)
        nearest_ennemy_to_goal = self.get_nearest_enemy_in_triangle(red_team if self.team==1 else blue_team)
        # Trouver les alliés et ennemis les plus proches
        if self.team == 0:  # Rouge
            nearest_ally = self.get_nearest_player(red_team)
            nearest_enemy = self.get_nearest_player(blue_team)
            goal = WIDTH - 50 
        else:  # Bleu
            nearest_ally = self.get_nearest_player(blue_team)
            nearest_enemy = self.get_nearest_player(red_team)
            goal = 50
        #pygame.draw.line(screen, ORANGE, (self.x, self.y ), (goal, HEIGHT / 2 ), 2)
        #pygame.display.update()    
        #pygame.draw.line(screen, (255, 0, 0), (self.x, self.y), (nearest_ally.x, nearest_ally.y), 2)
        #pygame.draw.line(screen, (255, 0, 0), (self.x, self.y), (nearest_enemy.x, nearest_enemy.y), 2)
        #pygame.display.update()
        # Angle et distance vers les alliés et ennemis
        angle_ally = self.angle_between_orientation_and_segment(nearest_ally) if nearest_ally else 0
        distance_ally = math.hypot(self.x - nearest_ally.x, self.y - nearest_ally.y) if nearest_ally else float('inf')
        angle_enemy = self.angle_between_orientation_and_segment(nearest_enemy) if nearest_enemy else 0
        distance_enemy = math.hypot(self.x - nearest_enemy.x, self.y - nearest_enemy.y) if nearest_enemy else float('inf')
        angle_enbut = self.angle_between_orientation_and_segment(blue_goal) if self.team == 1 else self.angle_between_orientation_and_segment(red_goal)
        distance_goal = math.hypot(self.x - blue_goal.x, self.y - blue_goal.y) / math.hypot(WIDTH, HEIGHT / 2) if self.team == 1 else math.hypot(self.x - red_goal.x, self.y - red_goal.y)  / math.hypot(WIDTH, HEIGHT / 2)
        relatif_x = self.x / WIDTH if self.team == 0 else (WIDTH - self.x) / WIDTH
        relatif_y = (self.y / HEIGHT * 2) - 1
        #distance_goal = WIDTH 
        # Statut de la balle
        if ball.holder:
            ball_status = 1 if ball.holder.team == self.team else -1
        else:
            ball_status = 0

        # Possession de la balle par le joueur actuel

        has_ball = 1 if self.has_ball else 0

        # Préparer les entrées pour le réseau neuronal
        inputs = np.array([
            angle_ball / math.pi, distance_ball / math.hypot(WIDTH, HEIGHT),
            angle_ally / math.pi, distance_ally / math.hypot(WIDTH, HEIGHT),
            angle_enemy / math.pi, distance_enemy / math.hypot(WIDTH, HEIGHT),
            ball_status, has_ball, #relatif_x,
            relatif_y if self.team==0 else relatif_y * -1,
            angle_enbut / math.pi , distance_goal,
            self.get_opponents_above_ratio(opponents) if self.team==0 else self.get_opponents_above_ratio(opponents) * -1, nearest_ennemy_to_goal/WIDTH # Ajout du ratio des adversaires
        ])

        if selected_player==self:
            selected_player_input=inputs
      
        # Passer les entrées dans le réseau neuronal
        outputs = self.nn.forward(inputs)

        # Interprétation des sorties
        rotation = -1 if outputs[0] < -0.5 else 1 if outputs[0] > 0.5 else 0
        make_pass = outputs[1] > 0.5
        boost = outputs [2] > 0.5
        if make_pass and self.has_ball:
         allies_list = red_team if self.team == 0 else blue_team
         self.pass_to_ally(ball, allies_list)

        if boost:
            self.activate_boost()

        return rotation, make_pass, boost
    
    def evaluate_action(self, prev_ball_position, ball_position,  prev_distance_to_ball):
        reward = 0
        if self.team == 0:  # Rouge
            nearest_ally = self.get_nearest_player(red_team)
            nearest_enemy = self.get_nearest_player(blue_team)
            goal = WIDTH - 50 
        else:  # Bleu
            nearest_ally = self.get_nearest_player(blue_team)
            nearest_enemy = self.get_nearest_player(red_team)
            goal = 50
        
        distance_enemy = math.hypot(self.x - nearest_enemy.x, self.y - nearest_enemy.y) if nearest_enemy else float('inf')

        if self.team == 0:
           own_goal = 0
           opponent_goal = WIDTH
        else:
           own_goal = WIDTH
           opponent_goal = 0
        
        if ball.holder:
            ball_status = 1 if ball.holder.team == self.team else -1
        else:
            ball_status = 0

        # 1. S'approcher de la balle
        current_distance_to_ball = math.hypot(self.x - ball_position[0], self.y - ball_position[1])
        if current_distance_to_ball < prev_distance_to_ball:
            reward += 20000 
        else:
            reward -= 50000 
        
        # si l'equipe a la balle etre pret du porteur c'est bien mais pas trop pret
        if ball_status == 1 and 50 < current_distance_to_ball < 150 and ball.holder is not self:
            reward += 10000

        if ball_status==-1 and current_distance_to_ball <  250 and (distance_enemy > 10 or nearest_enemy.has_ball==True):
            reward += 10000


        # ne pas se bloquer dans les bord
        if self.x < 40 or self.x > WIDTH -40 or self.y < 50 or self.y > HEIGHT -50:
           reward -= 2000 


        # 2. Rester entre la balle et la ligne d'essai adverse
        #if  self.x < ball_position[0]  if self.team == 0 else self.x > ball_position[0]:
         #  reward += 10
        #else:
          #  reward -= 100

        # 3. Ramasser la balle
        if self.has_ball:
            reward += 500
        
        # 3.2 faire une passe

        # 4. Approcher la balle de l'en-but adverse
        if self.has_ball == True:
          prev_ball_to_goal_distance = abs(prev_ball_position[0] - opponent_goal)
          current_ball_to_goal_distance = abs(ball_position[0] - opponent_goal)
          if current_ball_to_goal_distance < prev_ball_to_goal_distance:
            reward += 100000
          elif current_ball_to_goal_distance > prev_ball_to_goal_distance:
            reward -= 200000
        
        #punition collecitve 
        factor_puni = 2 if ball_status == 1 else 1
        if self.team == 0:
            reward += ball_position[1] - WIDTH / 2 * factor_puni
        else:
            reward -= ball_position[1] - WIDTH / 2 * factor_puni
 

        #  5. il faut bouger
        distance_point_depard = math.hypot(self.x - self.intial_x, self.y - self.intial_y)
        if distance_point_depard > 250:
            reward +=30
        else:
            reward -=1000

        return reward

    def is_clicked(self, mouse_x, mouse_y):
        """Vérifie si un clic de souris est sur le joueur."""
        return math.hypot(self.x - mouse_x, self.y - mouse_y) < 10
    
    def update(self, ball):
        # Sauvegarder la position précédente
        current_time = time.time()
        prev_x, prev_y = self.x, self.y
        rotation, make_pass, boost = self.decide(ball)

        # Appliquer la rotation
        self.orientation += rotation * self.rotation_speed
        self.orientation %= 2 * math.pi
        if self.boosting and current_time >= self.boost_end_time:
            self.boosting = False
            self.speed -= 2  # Restaurer la vitesse normale
            #print(f"Boost terminé pour le joueur {self} à {current_time:.2f} secondes.")

        # Avancer dans la direction actuelle
        self.x += math.cos(self.orientation) * self.speed
        self.y += math.sin(self.orientation) * self.speed
        
        # Contraindre la position dans les limites du terrain
        self.constrain_position()
        
        # Détecter les collisions
        other_players = [player for player in red_team + blue_team if player != self]
        if self.detect_collision(other_players):
        # Revenir à la position précédente en cas de collision
          self.x, self.y = prev_x, prev_y

 
        # Ramasser la balle si possible
        self.pick_up_ball(ball)

        # Vérifier si le joueur marque
        self.check_score_and_reset(ball, WIDTH, HEIGHT, score, red_team + blue_team )

    def draw(self, screen):
        
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), 10)
        # Dessiner la génération dans le cercle
        font = pygame.font.Font(None, 24)  # Taille 24 pour la génération
        text_surface = font.render(str(self.generation), True, (255, 255, 255))  # Texte blanc
        text_rect = text_surface.get_rect(center=(self.x, self.y))  # Centrer le texte
        screen.blit(text_surface, text_rect)
        if self == selected_player:
           pygame.draw.circle(
             screen, 
             (255, 255, 0),  # Couleur jaune
             (int(self.x), int(self.y)),  # Position du joueur
             20,  # Rayon du cercle (ajustez selon vos besoins)
             2    # Épaisseur du cercle (2 pixels)
    )
    
    def to_dict(self):
    # Convertir les données du joueur en dictionnaire
        return {
         "x": self.x,
         "y": self.y,
         "orientation": self.orientation,
         "team": self.team,
         "has_ball": self.has_ball,
         "profile": self.profile,  # Correction du nom "porfile" en "profile"
         "reward_total": self.reward_total,
         "nn_weights1": self.nn.weights1.tolist() if isinstance(self.nn.weights1, np.ndarray) else self.nn.weights1,
         "nn_weights2": self.nn.weights2.tolist() if isinstance(self.nn.weights2, np.ndarray) else self.nn.weights2,
        }

    def detect_collision(self, other_players, player_radius=10):
       """
        Détecte les collisions avec les autres joueurs ou les limites du terrain,
       et repousse légèrement les joueurs en cas de collision.

       :param other_players: Liste des autres joueurs sur le terrain.
       :param player_radius: Rayon des joueurs (par défaut : 10).
       """
       for other in other_players:
            if other is not self:  # Ne pas se comparer à soi-même
                distance = math.hypot(self.x - other.x, self.y - other.y)
                if distance < 2 * player_radius:  # Si les joueurs se touchent
                # Calcul de la direction de repousser le joueur
                   dx = self.x - other.x
                   dy = self.y - other.y
                   angle = math.atan2(dy, dx)  # Angle de collision
                   diff_force = (self.force - other.force+2.5)  

                   # Repousser légèrement dans la direction opposée
                   repel_distance = 4  # Distance de "repoussement"
                   self.x += math.cos(self.orientation) * repel_distance / diff_force
                   self.y += math.sin(self.orientation) * repel_distance / diff_force
                   if self.team!=other.team:
                      #self.reward_total += 1000 * diff_force
                      if other.has_ball==True:
                         other.drop_ball(ball)
                         self.reward_total+=5000
                   other.x -= math.cos(other.orientation) * repel_distance * diff_force
                   other.y -= math.sin(other.orientation) * repel_distance * diff_force
                   

                   return True  # Collision détectée
       return False  # Aucune collision détectée
    
    def save(self, filename="players_data.json"):
     # Initialiser une liste de données
     data = []

     # Vérifier si le fichier existe et lire son contenu
     if os.path.exists(filename):
        with open(filename, 'r') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):  # Si ce n'est pas une liste, réinitialiser
                    data = []
            except json.JSONDecodeError:
                data = []  # Si le fichier est vide ou corrompu, on recommence avec une liste

     # Ajouter les données du joueur actuel
     data.append(self.to_dict())

     # Réécrire les données dans le fichier
     with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    def get_farthest_ally_within_radius(self, team_players, radius=150):
      """
      Trouve l'allié le plus éloigné dans un rayon donné.
     
     :param team_players: Liste des joueurs de la même équipe (alliés).
     :param radius: Rayon maximum pour rechercher les alliés (par défaut 250).
     :return: L'allié le plus éloigné (objet Player) ou None s'il n'y en a pas.
     """
      farthest_ally = None
      max_distance = 0
      
      for ally in team_players:
        if ally is not self and ally.x > self.x if self.team == 1 else ally.x < self.x:  # Ne pas considérer soi-même
            distance = math.hypot(self.x - ally.x, self.y - ally.y)
            if distance <= radius and distance > max_distance:
                max_distance = distance
                farthest_ally = ally
        #if farthest_ally:          
         #pygame.draw.line(screen, ORANGE, (self.x, self.y), (farthest_ally.x, farthest_ally.y), 2)
         #pygame.display.update()

      return farthest_ally
    
    def pass_to_ally(self, ball, allies):
     """
     Passe la balle à l'allié le plus proche.
    
     :param ball: L'objet balle.
     :param allies: Liste des alliés.
     """
     fareth_ally = self.get_farthest_ally_within_radius(allies)
     if self.has_ball and fareth_ally and (self.x > 200 if self.team == 0 else self.x < WIDTH -200):  # Le joueur doit posséder la balle
            # Définir la nouvelle position de la balle pour simuler la passe
            #ball.x = fareth_ally.x
            #ball.y = fareth_ally.y
            ball.holder = None  # Attribue la possession à l'allié
            ball.target_x = fareth_ally.x
            ball.target_y = fareth_ally.y
            ball.speed = 6
            self.drop_ball(ball)  # Le joueur actuel lâche la balle
            #fareth_ally.has_ball = True  # L'allié reçoit la balle
            self.reward_total += 100
            pygame.draw.line(screen, ORANGE, (self.x, self.y), (fareth_ally.x, fareth_ally.y), 2)
            pygame.display.update()
    
    @classmethod
    def load(cls, filepath):
        """Charge les poids du réseau neuronal et le score depuis un fichier JSON."""
        with open(filepath, "r") as f:
            player_data = json.load(f)
        player = cls(0, 0, 0, 0)  # Position par défaut pour charger
        player.reward_total = player_data["score"]
        player.nn.weights = np.array(player_data["neural_network"]["weights"])
        player.nn.biases = np.array(player_data["neural_network"]["biases"])
        return player
    
    @classmethod
    def load_players(cls, filename="players_data.json"):
        players = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                try:
                    data = json.load(f)
                    for player_data in data:
                        player = cls.from_dict(player_data)
                        players.append(player)
                except json.JSONDecodeError:
                    print("Erreur : Fichier JSON corrompu.")
        return players

    @classmethod
    def from_dict(cls, data):
        """Créer un joueur à partir d'un dictionnaire."""
        player = cls(
            x=data["x"],
            y=data["y"],
            orientation=data["orientation"],
            team=data["team"]
        )
        player.has_ball = data["has_ball"]
        player.reward_total = data["reward_total"]
        player.nn.weights1 = np.array(data["nn_weights1"])
        player.nn.weights2 = np.array(data["nn_weights2"])
        return player

# Classe NeuralNetwork
class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        """
        Initialise le réseau neuronal avec deux couches cachées.
        :param input_size: Nombre d'entrées.
        :param hidden1_size: Nombre de neurones dans la première couche cachée.
        :param hidden2_size: Nombre de neurones dans la deuxième couche cachée.
        :param output_size: Nombre de sorties.
        """
        self.weights1 = np.random.randn(input_size, hidden1_size)
        self.weights2 = np.random.randn(hidden1_size, hidden2_size)
        self.weights3 = np.random.randn(hidden2_size, output_size)

        # Vérification que les poids sont des ndarrays
        assert isinstance(self.weights1, np.ndarray), "weights1 n'est pas un numpy.ndarray"
        assert isinstance(self.weights2, np.ndarray), "weights2 n'est pas un numpy.ndarray"
        assert isinstance(self.weights3, np.ndarray), "weights3 n'est pas un numpy.ndarray"

    def forward(self, inputs):
        """
        Calcule la sortie du réseau neuronal en utilisant les poids.
        :param inputs: Les entrées du réseau.
        :return: La sortie finale.
        """
        # Première couche cachée
        hidden1 = np.dot(inputs, self.weights1)
        hidden1 = np.tanh(hidden1)

        # Deuxième couche cachée
        hidden2 = np.dot(hidden1, self.weights2)
        hidden2 = np.tanh(hidden2)

        # Couche de sortie
        output = np.dot(hidden2, self.weights3)
        return np.tanh(output)

    def calculate_activations(self, inputs):
        """
        Calcule les activations des deux couches cachées et de la couche de sortie.
        :param inputs: Les entrées du réseau.
        :return: Les activations des couches (hidden1, hidden2, output).
        """
        # Première couche cachée
        hidden1_raw = np.dot(inputs, self.weights1)
        hidden1_activations = np.tanh(hidden1_raw)

        # Deuxième couche cachée
        hidden2_raw = np.dot(hidden1_activations, self.weights2)
        hidden2_activations = np.tanh(hidden2_raw)

        # Couche de sortie
        output_raw = np.dot(hidden2_activations, self.weights3)
        output_activations = np.tanh(output_raw)

        # Retourner toutes les activations
        return hidden1_activations, hidden2_activations, output_activations
   
def load_joueur(json_file, position):
    """
    Charge un joueur depuis un fichier JSON et le place à la position spécifiée.

        :param json_file: Le chemin vers le fichier JSON contenant les données du joueur.
        :param position: Tuple (x, y) représentant la position du joueur sur le terrain.
        :return: Un objet Player initialisé avec les données du fichier JSON.
        """
        # Charger les données depuis le fichier JSON
    with open(json_file, "r") as f:
       players_data = json.load(f)

    # Choisir le joueur correspondant, ici on prend simplement le premier joueur du fichier
    player_data = players_data[0]  # Tu peu        x aussi ajouter un paramètre pour choisir un joueur spécifique

    # Extraire les données du joueur
    x = position[0]
    y = position[1]
    orientation = player_data["orientation"]
    team = player_data["team"]
    has_ball = player_data["has_ball"]
    profile = player_data["profile"]
    reward_total = player_data["reward_total"]
    nn_weights1 = player_data["nn_weights1"]
    nn_weights2 = player_data["nn_weights2"]

    # Créer un réseau neuronal avec les poids extraits du JSON
    nn = NeuralNetwork(len(nn_weights1), len(nn_weights1[0]), len(nn_weights2[0]))
    nn.weights1 = np.array(nn_weights1)
    nn.weights2 = np.array(nn_weights2)
    #        Créer un nouvel objet Player avec ces données
    player = Player(x, y, orientation, team, speed=player_data.get("speed", 2), force=player_data.get("force", 2), nn=nn, generation=player_data.get("generation", 1), profile=profile)

    return player


def crossover_nn(nn1, nn2, mutation_rate=0.1):
    """
    Combine les poids de deux réseaux neuronaux pour créer un nouveau réseau,
    avec une mutation aléatoire des poids.

    :param nn1: Réseau neuronal du premier parent.
    :param nn2: Réseau neuronal du deuxième parent.
    :param mutation_rate: Taux de mutation pour perturber les poids.
    :return: Nouveau réseau neuronal issu du croisement.
    """
    # Déterminer les dimensions des couches
    input_size = len(nn1.weights1)
    hidden_size = len(nn1.weights1[0])
    output_size = len(nn1.weights2[0])

    # Créer un nouveau réseau neuronal
    new_nn = NeuralNetwork(input_size, hidden_size, output_size)

    # Croisement et mutation pour weights1
    new_nn.weights1 = [
        [
            (nn1.weights1[i][j] if random.random() < 0.5 else nn2.weights1[i][j]) +
            (random.uniform(-mutation_rate, mutation_rate) if random.random() < 0.1 else 0)
            for j in range(hidden_size)
        ]
        for i in range(input_size)
    ]

    # Croisement et mutation pour weights2
    new_nn.weights2 = [
        [
            (nn1.weights2[i][j] if random.random() < 0.5 else nn2.weights2[i][j]) +
            (random.uniform(-mutation_rate, mutation_rate) if random.random() < 0.1 else 0)
            for j in range(output_size)
        ]
        for i in range(hidden_size)
    ]

    return new_nn
    
    
    # Mélange des poids et biais du deuxième layer
    new_nn.weights2 = np.where(
        np.random.rand(*nn1.weights2.shape) < 0.5,
        nn1.weights2,
        nn2.weights2
    )
    
    
    return new_nn


def calculate_orientation_to_target(self, target_x, target_y):
    """
    Calcule l'orientation nécessaire pour diriger une passe vers une cible donnée.
    
    :param target_x: Coordonnée x de la cible.
    :param target_y: Coordonnée y de la cible.
    :return: Orientation en radians.
    """
    dx = target_x - self.x  # Différence en x
    dy = target_y - self.y  # Différence en y
    return math.atan2(dy, dx)  # Retourne l'angle en radians

def generate_team(team_num, num_players, side):
    team = []
    for _ in range(num_players):
        if side == "left":  # Demi-terrain gauche pour l'équipe rouge
            x = random.uniform(0, WIDTH / 2)
        else:  # Demi-terrain droit pour l'équipe bleue
            x = random.uniform(WIDTH / 2, WIDTH)
        
        y = random.uniform(0, HEIGHT)  # Position y sur toute la hauteur
        orientation = 0 if side == "left" else math.pi  # Vers l'avant pour chaque camp
        team.append(Player(x, y, orientation, team_num))
    return team

def select_top_players(players, percentage=0.5):
    players.sort(key=lambda p: p.reward_total, reverse=True)  # Trier par reward_total
    cutoff = int(len(players) * percentage)  # Calculer le seuil des meilleurs
    return players[:cutoff]  # Retourner les meilleurs joueurs

def mutate_nn(nn, mutation_rate=0.1):
    mutated_nn = NeuralNetwork(13, 26, 20, 3)  # Créer un nouveau NN
    mutated_nn.weights1 = [
        [weight + random.uniform(-mutation_rate, mutation_rate) if random.random() < 0.1 else weight
         for weight in layer]
        for layer in nn.weights1
    ]
    mutated_nn.weights2 = [
        [weight + random.uniform(-mutation_rate, mutation_rate) if random.random() < 0.1 else weight
         for weight in layer]
        for layer in nn.weights2
    ]
    mutated_nn.weights3 = [
        [weight + random.uniform(-mutation_rate, mutation_rate) if random.random() < 0.1 else weight
         for weight in layer]
        for layer in nn.weights3
    ]
    return mutated_nn

def select_parent(top_players):
    """
    Sélectionne un parent en fonction de la probabilité définie :
    - 30% pour les 10% meilleurs joueurs.
    - 20% pour les 15% suivants.
    - 50% pour les 75% restants.
    
    :param top_players: Liste triée des joueurs (du meilleur au moins bon).
    :return: Un joueur choisi en fonction des probabilités.
    """
    total_players = len(top_players)
    
    # Diviser les joueurs en groupes
    top_10_percent = top_players[:max(1, total_players // 10)]
    next_15_percent = top_players[max(1, total_players // 10):max(1, total_players // 10 + total_players // 15)]
    remaining_75_percent = top_players[max(1, total_players // 10 + total_players // 15):]

    # Choisir un groupe basé sur les probabilités
    r = random.random()
    if r < 0.3:  # 30% chance
        return random.choice(top_10_percent)
    elif r < 0.5:  # 20% chance
        return random.choice(next_15_percent)
    else:  # 50% chance
        return random.choice(remaining_75_percent)

def generate_new_players(top_players, total_players):
    """
    Génère de nouveaux joueurs en combinant mutation et croisement des meilleurs joueurs.

    :param top_players: Liste triée des meilleurs joueurs (du meilleur au moins bon).
    :param total_players: Nombre total de joueurs souhaité.
    :return: Liste des nouveaux joueurs.
    """
    new_players = []
    sorted_players = sorted(red_team + blue_team, key=lambda p: p.reward_total, reverse=True)

    # Générer des joueurs par mutation des joueurs existants
    weakest_players = sorted_players[total_players // 2:]
    for player_faible in weakest_players:
        parent = select_parent(top_players)
        mutated_nn = mutate_nn(parent.nn)
        new_player = Player(
            x=random.randint(50, WIDTH - 50),
            y=random.randint(50, HEIGHT - 50),
            orientation=random.uniform(0, 2 * math.pi),
            team=player_faible.team,
            nn=mutated_nn,
            generation=parent.generation + 1,
            profile=parent.profile,
            speed=parent.speed,
            force=parent.force
        )
        new_players.append(new_player)
        new_player.change_color()

    return top_players + new_players
    
def reset_game(players, ball):
    # Réinitialiser la position des joueurs
    for player in players:
        player.x = random.randint(50, WIDTH / 2 - 50) if player.team == 0 else random.randint(WIDTH / 2, WIDTH - 50 / 2 - 50)
        player.y = random.randint(50, HEIGHT - 50)
        player.reward_total = 0
        player.has_ball = False
        player.cooldown = 0  # Réinitialisation des cooldowns
        

    # Réinitialiser la position du ballon
    ball.x = WIDTH / 2
    ball.y = HEIGHT / 2
    ball.holder = None

def count_profiles(players):
    """
    Compte le nombre de joueurs par profil dans une liste de joueurs.

    :param players: Liste des joueurs.
    :return: Dictionnaire contenant le nombre de chaque profil.
    """
    profile_counts = {"bloqueur": 0, "linemen": 0, "courreur": 0}
    for player in players:
        if player.profile in profile_counts:
            profile_counts[player.profile] += 1
    return profile_counts

def draw_profile_table(screen, red_team, blue_team, font, x, y):
    """
    Affiche un tableau avec le nombre de joueurs de chaque profil par équipe.

    :param screen: Écran Pygame sur lequel dessiner.
    :param red_team: Liste des joueurs de l'équipe rouge.
    :param blue_team: Liste des joueurs de l'équipe bleue.
    :param font: Police à utiliser pour l'affichage.
    :param x: Position x du tableau.
    :param y: Position y du tableau.
    """
    # Compter les profils pour chaque équipe
    red_profiles = count_profiles(red_team)
    blue_profiles = count_profiles(blue_team)

    # Titre du tableau
    title_text = font.render("Profils des joueurs", True, WHITE)
    screen.blit(title_text, (x, y))

    # Dessiner les colonnes
    headers = ["Profil", "Rouge", "Bleu"]
    header_y = y + 30
    for i, header in enumerate(headers):
        header_text = font.render(header, True, (0,0,0))
        screen.blit(header_text, (x + i * 120, header_y))

    # Afficher les données
    profiles = ["bloqueur", "linemen", "courreur"]
    for i, profile in enumerate(profiles):
        profile_text = font.render(profile.capitalize(), True, (0,0,0))
        red_count_text = font.render(str(red_profiles[profile]), True, RED)
        blue_count_text = font.render(str(blue_profiles[profile]), True, BLUE)

        row_y = header_y + (i + 1) * 30
        screen.blit(profile_text, (x, row_y))
        screen.blit(red_count_text, (x + 120, row_y))
        screen.blit(blue_count_text, (x + 240, row_y))

def draw_field():
    screen.fill(GREEN)
    pygame.draw.rect(screen, WHITE, (50, 50, WIDTH - 100 , HEIGHT - 100), 5)
    pygame.draw.line(screen, WHITE, (WIDTH / 2, 50), (WIDTH / 2 , HEIGHT -50 ), 5)
    pygame.draw.circle(screen, WHITE, (400, 200), 50, 5)
    pygame.draw.rect(screen, (255, 255, 255), (WIDTH, 0, 500, HEIGHT))

def draw_histogram(screen, players, position=(WIDTH, 10), size=(300, 500)):
    """Dessine un histogramme de la population par génération.
    
    Args:
        screen: Surface Pygame où dessiner l'histogramme.
        players: Liste des joueurs (red_team + blue_team).
        position: Position du coin supérieur gauche de l'histogramme.
        size: Taille (largeur, hauteur) de l'histogramme.
    """
    # Dimensions de l'histogramme
    x, y = position
    width, height = size

    # Compter la population par génération
    generation_counts = {}
    for player in players:
        generation = player.generation
        generation_counts[generation] = generation_counts.get(generation, 0) + 1

    # Normaliser les données pour l'affichage
    max_population = max(generation_counts.values(), default=1)
    num_generations = len(generation_counts)

    # Largeur d'une barre
    bar_width = width // max(1, num_generations)

    # Dessiner les barres
    for i, (generation, count) in enumerate(sorted(generation_counts.items())):
        bar_height = int((count / max_population) * height)
        bar_x = x + i * bar_width
        bar_y = y + height - bar_height

        # Dessiner la barre
        pygame.draw.rect(screen, (100, 200, 255), (bar_x, bar_y, bar_width - 2, bar_height))

        # Ajouter le numéro de la génération
        font = pygame.font.Font(None, 20)
        text_surface = font.render(str(generation), True, (0,0,0))
        text_rect = text_surface.get_rect(center=(bar_x + bar_width // 2, y + height + 10))
        screen.blit(text_surface, text_rect)

def draw_score(screen, score, font, position=(10, 10)):
    """Affiche le score des équipes sur l'écran.

    Args:
        screen: Surface Pygame où dessiner le texte.
        score: Dictionnaire contenant les scores des équipes.
        font: Police utilisée pour dessiner le texte.
        position: Position (x, y) du coin supérieur gauche du texte.
    """
    red_score = score.get('red', 0)
    blue_score = score.get('blue', 0)

    # Texte du score
    score_text = f"Rouge: {red_score} - Bleu: {blue_score}  genearation : {current_round }/30"
    text_surface = font.render(score_text, True, (255, 255, 255))  # Blanc

    # Dessiner le texte à la position spécifiée
    screen.blit(text_surface, position)

def draw_neural_network(screen, nn, x, y, width, height, inputs):
    """
    Dessine un réseau neuronal avec deux couches cachées.
    :param screen: Surface Pygame.
    :param nn: Réseau neuronal (instance de NeuralNetwork).
    :param x: Coordonnée x du coin supérieur gauche.
    :param y: Coordonnée y du coin supérieur gauche.
    :param width: Largeur du dessin.
    :param height: Hauteur du dessin.
    :param inputs: Liste des entrées du réseau neuronal.
    """
    # Calcul des activations
    hidden1_activations, hidden2_activations, output_activations = nn.calculate_activations(inputs)

    # Dimensions des couches
    input_size = len(inputs)
    hidden1_size = len(hidden1_activations)
    hidden2_size = len(hidden2_activations)
    output_size = len(output_activations)

    # Positions des neurones
    input_positions = [(x, y + i * (height // (input_size + 1))) for i in range(input_size)]
    hidden1_positions = [(x + width // 3, y + i * (height // (hidden1_size + 1))) for i in range(hidden1_size)]
    hidden2_positions = [(x + 2 * width // 3, y + i * (height // (hidden2_size + 1))) for i in range(hidden2_size)]
    output_positions = [(x + width, y + i * (height // (output_size + 1))) for i in range(output_size)]

    # Dessiner les connexions entre les couches
    # Entrées -> Première couche cachée
    for i, input_pos in enumerate(input_positions):
        for j, hidden1_pos in enumerate(hidden1_positions):
            weight = nn.weights1[i][j]
            color = (0, 255, 0) if weight > 0 else (255, 0, 0)
            pygame.draw.line(screen, color, input_pos, hidden1_pos, max(1, int(abs(weight) * 2)))

    # Première couche cachée -> Deuxième couche cachée
    for i, hidden1_pos in enumerate(hidden1_positions):
        for j, hidden2_pos in enumerate(hidden2_positions):
            weight = nn.weights2[i][j]
            color = (0, 255, 0) if weight > 0 else (255, 0, 0)
            pygame.draw.line(screen, color, hidden1_pos, hidden2_pos, max(1, int(abs(weight) * 2)))

    # Deuxième couche cachée -> Sorties
    for i, hidden2_pos in enumerate(hidden2_positions):
        for j, output_pos in enumerate(output_positions):
            weight = nn.weights3[i][j]
            color = (0, 255, 0) if weight > 0 else (255, 0, 0)
            pygame.draw.line(screen, color, hidden2_pos, output_pos, max(1, int(abs(weight) * 2)))

    # Dessiner les neurones avec intensité basée sur l'activation
      # Dessiner les neurones avec intensité basée sur l'activation
    for idx, pos in enumerate(input_positions):
        if inputs is not None:
            # Changer la couleur en fonction de la valeur d'entrée
            value = inputs[idx]
            if value == float('inf') or value == float('-inf'):
                intensity = 0  # Remplacer inf par 0 ou une valeur par défaut
            else:
                intensity = int((value + 1) / 2 * 255)  # Normaliser la valeur entre 0 et 255
            color = (intensity, intensity, 0)  # Couleur jaune avec intensité
        else:
            color = BLUE_B  # Couleur par défaut si pas d'entrées

        pygame.draw.circle(screen, color, pos, 5)

    for i, pos in enumerate(hidden1_positions):
        activation_intensity = int(255 * abs(hidden1_activations[i]))
        color = (activation_intensity, 0, 0) if hidden1_activations[i] < 0 else (0, activation_intensity, 0)
        pygame.draw.circle(screen, color, pos, 5)

    for i, pos in enumerate(hidden2_positions):
        activation_intensity = int(255 * abs(hidden2_activations[i]))
        color = (activation_intensity, 0, 0) if hidden2_activations[i] < 0 else (0, activation_intensity, 0)
        pygame.draw.circle(screen, color, pos, 5)

    for i, pos in enumerate(output_positions):
        activation_intensity = int(255 * abs(output_activations[i]))
        color = (activation_intensity, 0, 0) if output_activations[i] < 0 else (0, activation_intensity, 0)
        pygame.draw.circle(screen, color, pos, 5)

# Initialisation de Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH+400, HEIGHT))
pygame.display.set_caption("Simulation de Rugby")
clock = pygame.time.Clock()
blue_goal = Goal(0, HEIGHT / 2, team=0)
red_goal = Goal(WIDTH, HEIGHT / 2, team=1)

# Fonction pour dessiner le terrain


# Création des équipes et de la balle
red_team = generate_team(0, 55, "left")
blue_team = generate_team(1, 55, "right")
ball = Ball(WIDTH // 2, HEIGHT // 2)
rounds = 60  # Nombre de cycles d'évolution
current_round = 0 
start_time = pygame.time.get_ticks()  # Temps de départ
game_duration = 30000  # 30 secondes
score = {
    'red': 0,   # Score initial de l'équipe rouge
    'blue': 0   # Score initial de l'équipe bleue
}
# Boucle principale
running = True
while running:
    elapsed_time = pygame.time.get_ticks() - start_time
       # Vérifiez si le temps est écoulé pour ce cycle
    if elapsed_time >= game_duration:
        current_round += 1
        players = red_team + blue_team
        # Fin du cycle, sélection et mutation
        top_players = select_top_players(players)
        players = generate_new_players(top_players, 110)
        reset_game(players, ball)  # Réinitialiser la partie

        print(f"Cycle {current_round }/30 terminé.  joueurs recréés.")
        # Effectuer l'évolution et recréer les équipes
        print(f"Round {current_round} terminé. Evolution des joueurs...")
        # players = evolutionary_game(players, ball)

        # Vérifiez si tous les rounds sont terminés
        if current_round >= rounds:
            print("Fin de toutes les parties.")
            for player in players:
                player.save("test.json")  # Sauvegarder chaque joueur
            running = False
            continue

        # Réinitialiser les positions et l'état du jeu
        red_team = [player for player in players if player.team == 0]
        blue_team = [player for player in players if player.team == 1]
        ball = Ball(WIDTH // 2, HEIGHT // 2)
        start_time = pygame.time.get_ticks()  # Redémarrer le temps
        continue

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Clic gauche
             mouse_x, mouse_y = event.pos
             for player in red_team + blue_team:
               if math.hypot(player.x - mouse_x, player.y - mouse_y) < 20:
                 selected_player = player
                 break
             if mouse_x > WIDTH:
                 selected_player = None
    
    prev_ball_position = (ball.x, ball.y)
    # Mise à jour
    
    ball.update_position()
    ball_position = (ball.x, ball.y)
    for player in red_team + blue_team: 
        previous_distance_balle = math.hypot(player.x - prev_ball_position[0], player.y - prev_ball_position[1])
        player.update(ball)
        player.reward_total += player.evaluate_action( prev_ball_position, ball_position, previous_distance_balle)
        player.check_immobility(ball, 1 / FPS)  # Assurez-vous de fournir dt (1 / FPS)

    font = pygame.font.Font(None, 15)
    # Affichage
    draw_field()
    for player in red_team + blue_team:
        player.draw(screen)
    if selected_player_input is not None and selected_player is not None:
        draw_neural_network(screen, selected_player.nn, WIDTH+50, 300, 300, 700, selected_player_input)
       
    ball.draw(screen)
    draw_profile_table(screen, red_team, blue_team, font, x=WIDTH, y=200)
    #y_offset = 0
    #for team in (red_team, blue_team):
    #    for i, player in enumerate(team):
    #       score_text = font.render(
    #           f"Joueur {i+1} ({'Rouge' if color == RED else 'Bleu'}) : {player.reward_total}",
    #           True, color
    #         )
    #       screen.blit(score_text, (10, y_offset))
    #       y_offset += 10
    players = red_team + blue_team
    draw_histogram(screen, players, position=(WIDTH, 10), size=(400, 200))
    font = pygame.font.Font(None, 36)  # Taille 36, utilisez une autre police si vous préférez
    #   Afficher le score
    draw_score(screen, score, font, position=(WIDTH / 2, 10))
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
