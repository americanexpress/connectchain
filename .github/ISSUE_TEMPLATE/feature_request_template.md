name: Feature Request
description: Create a request for a new feature or functionality
title: "[Feature]: "
labels: ["feature"]
projects: ["americanexpress/connectchain"]
assignees:
  - none
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to request a new feature!
  - type: input
    id: contact
    attributes:
      label: Contact Details
      description: How can we get in touch with you if we need more info?
      placeholder: ex. email@example.com
    validations:
      required: false
  - type: textarea
    id: feature-description
    attributes:
      label: Please describe the functionality you wish to see implemented.
      description: Specific packages, usecases, and/or examples are helpful.
      placeholder: Tell us what you want to see!
      value: "New feature!"
    validations:
      required: true
  - type: textarea
    id: usage-example
    attributes:
      label: Please provide a code snippet of how you would use this feature.
      description: Showing the how you intend to use this feature will help us understand your usecase.
      render: shell
  - type: textarea
    id: version
    attributes:
      label: Version
      description: What version of our software are you running?
    validations:
      required: true
  - type: checkboxes
    id: terms
    attributes:
      label: Contribution Guidelines
      description: By submitting this request, you agree to follow our [Contribution Guidelines](../../CONTRIBUTING.md)
      options:
        - label: I agree to follow this project's Contribution Guidelines
          required: true
  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this request, you agree to follow our [Code of Conduct](../../CODE_OF_CONDUCT.md)
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
