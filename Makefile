IMAGE=dk64:latest
TEST_NAME=test-gym-env
ROM_PATH=/Users/rajpaul/cs1470/projects/parent-dk64/deepkart64/gym_mupen64plus/ROMs

build:
	docker build -t $(IMAGE) .

destroy:
	docker container rm test-gym-env

verify: build
	docker run -it \
		--name $(TEST_NAME) \
		-p 5900 \
		--mount source="$(ROM_PATH)",target=/src/gym-mupen64plus/gym_mupen64plus/ROMs,type=bind \
		$(IMAGE) \
	# python /src/gym_mupen64plus/verifyEnv.py
	pwd