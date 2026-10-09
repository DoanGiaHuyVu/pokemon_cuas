"""Read-only native input readiness, using the same guard as real dispatch."""
import time
from computer.capture import CaptureError, MacWindowBackend, load_calibration
from computer.executor import ExecutionError, MacExecutionGuard, focus_game_window
from computer.geometry import GeometryError


def check_game_input(profile, *, focus=False, start_delay=3, backend=None, sleep=time.sleep):
    report={'ready':False,'screen_recording_granted':None,'accessibility_granted':None,
            'held_modifiers':[],'foreground_application':None,'reason':None,
            'game_inputs':0,'model_calls':0}
    try:
        import AppKit
        import Quartz
        sleep(start_delay)
        report['screen_recording_granted']=bool(Quartz.CGPreflightScreenCaptureAccess())
        report['accessibility_granted']=bool(Quartz.CGPreflightPostEventAccess())
        flags=Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateCombinedSessionState)
        for name,mask in [('Shift',Quartz.kCGEventFlagMaskShift),('Control',Quartz.kCGEventFlagMaskControl),
                          ('Option',Quartz.kCGEventFlagMaskAlternate),('Command',Quartz.kCGEventFlagMaskCommand),
                          ('Caps Lock',Quartz.kCGEventFlagMaskAlphaShift)]:
            if flags & mask:report['held_modifiers'].append(name)
        if not report['accessibility_granted']:
            raise ExecutionError('Accessibility permission is required for the app running Python. If this command runs in Terminal, enable Terminal in System Settings > Privacy & Security > Accessibility.')
        if not report['screen_recording_granted']:
            raise CaptureError('Screen Recording permission is required for the app running Python.')
        backend=backend or MacWindowBackend()
        calibration=load_calibration(profile)
        if not calibration.verified:raise ExecutionError('Calibration is not confirmed. Run --calibrate and --confirm-calibration.')
        if focus:focus_game_window(profile,backend=backend)
        active=AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        report['foreground_application']=str(active.localizedName()) if active is not None else None
        MacExecutionGuard(calibration.window,backend,calibration).prepare()
        report.update(ready=True,reason='Native input checks pass. No game keys or clicks were sent.')
    except (ExecutionError,CaptureError,GeometryError) as exc:
        report['reason']=str(exc)
    except (OSError,ValueError,ImportError):
        report['reason']='Could not run the native input check. Check the calibration file and local macOS runtime.'
    return report
