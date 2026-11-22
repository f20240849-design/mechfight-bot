"""
Real-time Projectile Dodging, Laser Beam Evasion, and Threat Warning System.
Computes 3D trajectory intersections, mortar splash predictions, and charging laser deflection.
"""
import math
from typing import Dict, List, Optional, Tuple, Any

class EvasionSystem:
    def __init__(self, mech_radius: float = 2.5, mech_height: float = 9.0):
        self.mech_radius = mech_radius
        self.mech_height = mech_height
        self.cylinder_buffer = 0.8  # Extra safety margin (meters)
        self.effective_radius = mech_radius + self.cylinder_buffer

    def evaluate(
        self,
        me: Dict[str, Any],
        obs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Processes visible projectiles, beam events, and charging enemies.
        Returns:
            {
                "threat_level": float,       # 0.0 (safe) to 1.0 (imminent hit)
                "dodge_vec": Optional[List[float]], # 2D unit vector to dodge horizontally
                "should_jump": bool,         # Jump to clear mortar splash / ground hazards
                "should_boost": bool,        # Boost immediately to escape blast/beam
                "incoming_damage": float,    # Estimated incoming burst damage
                "source": Optional[str]      # "gun", "mortar", "laser", etc.
            }
        """
        my_pos = me.get("pos", [0.0, 0.0, 0.0])
        my_vel = me.get("vel", [0.0, 0.0, 0.0])
        mx, my, mz = my_pos[0], my_pos[1], my_pos[2]
        mvx, mvy = my_vel[0], my_vel[1]

        threat_level = 0.0
        dodge_fx, dodge_fy = 0.0, 0.0
        should_jump = False
        should_boost = False
        incoming_damage = 0.0
        primary_source = None

        # -------------------------------------------------------------
        # 1. GUN & NPC PROJECTILES (Linear Raycast Closest-Approach)
        # -------------------------------------------------------------
        projectiles = obs.get("projectiles", [])
        for p in projectiles:
            p_kind = p.get("kind", "gun")
            px, py, pz = p.get("pos", [0.0, 0.0, 0.0])
            vx, vy, vz = p.get("vel", [0.0, 0.0, 0.0])

            if p_kind == "mortar":
                # Handled separately with parabolic physics below
                continue

            # Relative position & velocity (mech reference frame)
            rx = mx - px
            ry = my - py
            rz = (mz + 4.5) - pz  # Aimed at center of mass

            # Speed of projectile
            v_sq = vx * vx + vy * vy + vz * vz
            if v_sq < 1.0:
                continue

            # Time to closest approach: t = (r . v) / |v|^2
            t_ca = (rx * vx + ry * vy + rz * vz) / v_sq

            # Only care about incoming projectiles in the near future (within 0.55s)
            if 0.0 < t_ca < 0.55:
                # Closest point along bullet trajectory
                cpx = px + vx * t_ca
                cpy = py + vy * t_ca
                cpz = pz + vz * t_ca

                # Distance from our center axis
                dx = mx - cpx
                dy = my - cpy
                horiz_dist = math.hypot(dx, dy)
                vert_dist = abs((mz + 4.5) - cpz)

                if horiz_dist < self.effective_radius and vert_dist < 5.0:
                    # Imminent collision!
                    urgency = 1.0 - (t_ca / 0.55)
                    dmg = 25.0 if p_kind == "npc" else 45.0
                    incoming_damage += dmg

                    # Dodge perpendicular to bullet's horizontal direction
                    v_horiz = math.hypot(vx, vy)
                    if v_horiz > 1.0:
                        # Two perpendicular options: [-vy, vx] or [vy, -vx]
                        p1x, p1y = -vy / v_horiz, vx / v_horiz
                        p2x, p2y = vy / v_horiz, -vx / v_horiz

                        # Choose the perpendicular direction that moves AWAY from the trajectory
                        dot1 = p1x * dx + p1y * dy
                        dot2 = p2x * dx + p2y * dy
                        perp_x, perp_y = (p1x, p1y) if dot1 >= dot2 else (p2x, p2y)

                        # Weight dodge force by urgency
                        dodge_fx += perp_x * urgency * 2.0
                        dodge_fy += perp_y * urgency * 2.0

                        if urgency > threat_level:
                            threat_level = urgency
                            primary_source = "gun"
                            should_boost = True

        # -------------------------------------------------------------
        # 2. MORTAR SHELLS (Parabolic Ballistics & Splash Zone)
        # -------------------------------------------------------------
        for p in projectiles:
            if p.get("kind") != "mortar":
                continue

            px, py, pz = p.get("pos", [0.0, 0.0, 0.0])
            vx, vy, vz = p.get("vel", [0.0, 0.0, 0.0])
            g = 22.0  # Mortar gravity

            # Solve quadratic for ground impact: z(t) = pz + vz*t - 0.5*g*t^2 = mz
            # -0.5*g*t^2 + vz*t + (pz - mz) = 0
            # 11*t^2 - vz*t - (pz - mz) = 0
            a = 0.5 * g
            b = -vz
            c = -(pz - mz)

            disc = b * b - 4 * a * c
            if disc >= 0:
                sqrt_disc = math.sqrt(disc)
                t1 = (-b + sqrt_disc) / (2 * a)
                t2 = (-b - sqrt_disc) / (2 * a)
                t_impact = t1 if t1 > 0.05 else (t2 if t2 > 0.05 else -1.0)

                if 0.05 < t_impact < 2.5:
                    impact_x = px + vx * t_impact
                    impact_y = py + vy * t_impact

                    dist_to_impact = math.hypot(mx - impact_x, my - impact_y)
                    splash_radius = 8.5  # 7m splash + safety buffer

                    if dist_to_impact < splash_radius:
                        # In the splash zone!
                        urgency = 1.0 - (dist_to_impact / splash_radius)
                        incoming_damage += 200.0 * urgency

                        # Dodge directly away from splash center
                        if dist_to_impact > 0.1:
                            esc_x = (mx - impact_x) / dist_to_impact
                            esc_y = (my - impact_y) / dist_to_impact
                        else:
                            esc_x, esc_y = 1.0, 0.0

                        dodge_fx += esc_x * urgency * 3.0
                        dodge_fy += esc_y * urgency * 3.0

                        if urgency > threat_level:
                            threat_level = urgency
                            primary_source = "mortar"
                            should_boost = True

                        # Jump if impact is within 0.4s to reduce ground blast & energy drain
                        if t_impact < 0.45:
                            should_jump = True

        # -------------------------------------------------------------
        # 3. CHARGING LASER ENEMY DEFLECTION (Anticipatory Sidestep)
        # -------------------------------------------------------------
        mechs = obs.get("mechs", [])
        for m in mechs:
            if m.get("kind") != "mech":
                continue
            if not m.get("charging", False):
                continue

            epos = m.get("pos", [0.0, 0.0, 0.0])
            eaim = m.get("aim", [0.0, 0.0])
            eyaw, epitch = eaim[0], eaim[1]

            # Laser ray origin: enemy pos + muzzle height (7.0m)
            lx0 = epos[0]
            ly0 = epos[1]
            lz0 = epos[2] + 7.0

            # Ray unit vector
            cos_pitch = math.cos(epitch)
            ldx = cos_pitch * math.cos(eyaw)
            ldy = cos_pitch * math.sin(eyaw)
            ldz = math.sin(epitch)

            # Vector from laser origin to our center of mass
            vtox = mx - lx0
            vtoy = my - ly0
            vtoz = (mz + 4.5) - lz0

            # Projection along laser direction
            proj = vtox * ldx + vtoy * ldy + vtoz * ldz

            if 5.0 < proj < 280.0:  # Enemy is pointing towards us
                # Closest point on the laser beam line
                near_x = lx0 + ldx * proj
                near_y = ly0 + ldy * proj
                near_z = lz0 + ldz * proj

                d_beam = math.hypot(mx - near_x, my - near_y)
                d_beam_3d = math.hypot(d_beam, (mz + 4.5) - near_z)

                if d_beam < 5.0 and d_beam_3d < 6.5:
                    # Enemy's laser crosshair is directly on us or sweeping over us!
                    laser_urgency = 1.0 - (d_beam / 5.0)
                    incoming_damage += 315.0 * laser_urgency

                    # Sidestep perpendicular to enemy's aim line
                    perp1_x, perp1_y = -ldy, ldx
                    perp2_x, perp2_y = ldy, -ldx

                    # Choose side that takes us further from beam
                    dot1 = perp1_x * (mx - near_x) + perp1_y * (my - near_y)
                    dot2 = perp2_x * (mx - near_x) + perp2_y * (my - near_y)
                    step_x, step_y = (perp1_x, perp1_y) if dot1 >= dot2 else (perp2_x, perp2_y)

                    dodge_fx += step_x * laser_urgency * 4.0
                    dodge_fy += step_y * laser_urgency * 4.0

                    if laser_urgency > threat_level:
                        threat_level = max(threat_level, laser_urgency)
                        primary_source = "laser_charge"
                        should_boost = True

        # -------------------------------------------------------------
        # 4. FINALIZE DODGE VECTOR
        # -------------------------------------------------------------
        dodge_vec = None
        force_mag = math.hypot(dodge_fx, dodge_fy)
        if force_mag > 0.05 and threat_level > 0.15:
            dodge_vec = [dodge_fx / force_mag, dodge_fy / force_mag]

        return {
            "threat_level": min(1.0, threat_level),
            "dodge_vec": dodge_vec,
            "should_jump": should_jump,
            "should_boost": should_boost,
            "incoming_damage": incoming_damage,
            "source": primary_source
        }
